import torch
from torch.autograd import Variable
import torch.nn.functional as F

import onmt.translate.Beam
import onmt.io

from onmt.io.DatasetBase import PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD

# def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
#     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#     https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#         Args:
#             logits: logits distribution shape (..., vocabulary size)
#             top_k >0: keep only top k tokens with highest probability (top-k filtering).
#             top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#     """
#     top_k = min(top_k, logits.size(-1))  # Safety check
#     if top_k > 0:
#         # Remove all tokens with a probability less than the last token of the top-k
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         logits[indices_to_remove] = filter_value

#     if top_p > 0.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#         # Remove tokens with cumulative probability above the threshold
#         sorted_indices_to_remove = cumulative_probs >= top_p
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0

#         indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
#             dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
#         logits[indices_to_remove] = filter_value
#     return logits

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False, context = False):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.context = context
        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_choice(self, batch, data, beam = True):
        if beam:
            return translate_batch(batch, data)
        else:
            return random_sampling(batch, data)

    def random_sampling(self, batch, data, 
            min_length=0, 
            max_length=50, 
            sampling_temp=1.0,
            keep_topk=1,
            return_attention=False,
            ):
        #https://github.com/JasonBenn/duet/blob/master/generate.py

        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        src_vocabs = self.fields["tgt"].vocab

        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'gcn':
            _, src_lengths = batch.src
            # report_stats.n_src_words += src_lengths.sum()
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, \
            mask_in, mask_out, mask_loop, mask_sent = onmt.io.get_adj(batch)
            if hasattr(batch, 'morph'):
                morph, mask_morph = onmt.io.get_morph(batch)  # [b,t, max_morph]
            if hasattr(batch, 'ctx') and self.context:
                    context = onmt.io.make_features(batch, 'ctx')  # [b,t, max_morph]

        if data_type == 'gcn':
            # F-prop through the model.
            if hasattr(batch, 'morph'):
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                                       adj_arc_in, adj_arc_out, adj_lab_in,
                                       adj_lab_out, mask_in, mask_out,
                                       mask_loop, mask_sent, morph, mask_morph)
            else:
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                               adj_arc_in, adj_arc_out, adj_lab_in,
                               adj_lab_out, mask_in, mask_out,
                               mask_loop, mask_sent)
        else:
            enc_states, memory_bank = self.model.encoder(src, src_lengths)


        dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)
        memory_lengths = src_lengths
        src_map = batch.src_map if self.copy_attn else None

        # if isinstance(memory_bank, tuple):
        #     mb_device = memory_bank[0].device
        # else:
        #     mb_device = memory_bank.device
        mb_device = None
        block_ngram_repeat = 0

        def var(a): return Variable(a, volatile=True)
        def rvar(a): return var(a.repeat(1, 1, 1))

        if hasattr(batch, 'ctx') and self.context:
            context = onmt.io.make_features(batch, 'ctx')  # [b,t, max_morph]
            context_var = rvar(context.data)
        else:
            context_var = None

        # random_sampler = onmt.translate.RandomSampling(
        #     vocab.stoi[PAD_WORD],  vocab.stoi[BOS_WORD],  vocab.stoi[EOS_WORD],
        #     batch_size, mb_device, min_length, block_ngram_repeat,
        #     self._exclusion_idxs, return_attention, max_length,
        #     sampling_temp, keep_topk, memory_lengths)
        
        batch_offset=None
        alive_seq = torch.LongTensor(batch_size * 1, 1).fill_(vocab.stoi[BOS_WORD])
        alive_attn = None
        predictions = [[] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]
        attention = [[] for _ in range(batch_size)]
        original_batch_idx = torch.arange(batch_size)
        
        for step in range(max_length):
            # Shape: (1, B, 1)
            
            inp = var(alive_seq[:, -1].contiguous().view(1, -1, 1))
            
            if self.copy_attn:
                # Turn any copied words into UNKs.
                inp = inp.masked_fill(
                    inp.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
                )

            if self.context:
                dec_out, dec_states, dec_attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths, context = context_var)
            else:
                dec_out, dec_states, dec_attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths)

            if not self.copy_attn:
                if "std" in dec_attn:
                    attn = dec_attn["std"]
                else:
                    attn = None
                log_probs = self.model.generator(dec_out.squeeze(0))
                # returns [(batch_size x beam_size) , vocab ] when 1 step
                # or [ tgt_len, batch_size, vocab ] when full sentence
            else:
                attn = dec_attn["copy"]
                scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                              attn.view(-1, attn.size(2)),
                                              src_map)
                # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
                if batch_offset is None:
                    scores = scores.view(batch.batch_size, -1, scores.size(-1))
                else:
                    scores = scores.view(-1, self.beam_size, scores.size(-1))
                scores = collapse_copy_scores(
                    scores,
                    batch,
                    self._tgt_vocab,
                    src_vocabs,
                    batch_dim=0,
                    batch_offset=batch_offset
                )
                scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
                log_probs = scores.squeeze(0).log()
                #[(batch_size x beam_size) , vocab ] when 1 step

            log_probs = torch.div(log_probs, sampling_temp)
            logits = log_probs

            if sampling_temp == 0.0 or keep_topk == 1:
                # For temp=0.0, take the argmax to avoid divide-by-zero errors.
                # keep_topk=1 is also equivalent to argmax.
                topk_scores, topk_ids = logits.topk(1, dim=-1)
                if sampling_temp > 0:
                    topk_scores /= sampling_temp
            else:
                logits = torch.div(logits, sampling_temp)

                if keep_topk > 0:
                    top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
                    kth_best = top_values[:, -1].view([-1, 1])
                    kth_best = kth_best.repeat([1, logits.shape[1]]).float()

                    # Set all logits that are not in the top-k to -10000.
                    # This puts the probabilities close to 0.
                    ignore = torch.lt(logits, kth_best)
                    logits = logits.masked_fill(ignore, -10000)
            
            props = F.softmax(logits,dim = 1)
            topk_ids = torch.multinomial(props, 1)
            #topk_ids: Shaped ``(batch_size, 1)``.
            topk_scores = logits.gather(dim=1, index=topk_ids)
            #topk_scores: Shaped ``(batch_size, 1)``.

            is_finished = topk_ids.eq(vocab.stoi[EOS_WORD])
            topk_ids = torch.LongTensor(topk_ids.data)
            alive_seq = torch.cat([alive_seq, topk_ids], -1)
            if alive_attn is None:
                alive_attn = attn
            else:
                alive_attn = torch.cat([alive_attn, attn], 0)
    
            #
            any_batch_is_finished = is_finished.any()
            if any_batch_is_finished:
                #random_sampler.update_finished()
                finished_batches = is_finished.view(-1).nonzero()
                for b in finished_batches.view(-1):
                    b = int(b)
                    b_orig = original_batch_idx[b]
                    b_orig = int(b_orig)
                    scores[b_orig].append(float(topk_scores[b, 0].data))
                    predictions[b_orig].append(alive_seq[b, 1:].tolist())
                    attention[b_orig].append(
                        alive_attn[:, b, :memory_lengths[b]].data
                        if alive_attn is not None else [])
                done = is_finished.all()
                if done:
                    break

                #TODO remove done batches and train again

        ret = {}
        ret["scores"] = scores
        ret["predictions"] = predictions
        ret["attention"] = attention
        # return results
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'gcn':
            _, src_lengths = batch.src
            # report_stats.n_src_words += src_lengths.sum()
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, \
            mask_in, mask_out, mask_loop, mask_sent = onmt.io.get_adj(batch)
            if hasattr(batch, 'morph'):
                morph, mask_morph = onmt.io.get_morph(batch)  # [b,t, max_morph]
            if hasattr(batch, 'ctx') and self.context:
                    context = onmt.io.make_features(batch, 'ctx')  # [b,t, max_morph]

        if data_type == 'gcn':
            # F-prop through the model.
            if hasattr(batch, 'morph'):
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                                       adj_arc_in, adj_arc_out, adj_lab_in,
                                       adj_lab_out, mask_in, mask_out,
                                       mask_loop, mask_sent, morph, mask_morph)
            else:
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                               adj_arc_in, adj_arc_out, adj_lab_in,
                               adj_lab_out, mask_in, mask_out,
                               mask_loop, mask_sent)
        else:
            enc_states, memory_bank = self.model.encoder(src, src_lengths)


        dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if (data_type == 'text' or data_type == 'gcn') and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        if self.context:
            context_var = rvar(context.data)
            #context_var = var(torch.stack([b for b in context])
            #          .t().contiguous().view(1, -1))
        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            if self.context:
                dec_out, dec_states, attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths, context = context_var)
            else:
                dec_out, dec_states, attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'gcn':
            _, src_lengths = batch.src
            # report_stats.n_src_words += src_lengths.sum()
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, \
            mask_in, mask_out, mask_loop, mask_sent = onmt.io.get_adj(batch)
            if hasattr(batch, 'morph'):
                morph, mask_morph = onmt.io.get_morph(batch)  # [b,t, max_morph]
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        if hasattr(batch, 'ctx') and self.context:
                context = onmt.io.make_features(batch, 'ctx')  # [b,t, max_morph]

        #  (1) run the encoder on the src
        if data_type == 'gcn':
            # F-prop through the model.
            if hasattr(batch, 'morph'):
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                                       adj_arc_in, adj_arc_out, adj_lab_in,
                                       adj_lab_out, mask_in, mask_out,
                                       mask_loop, mask_sent, morph, mask_morph)
            else:
                enc_states, memory_bank = \
                    self.model.encoder(src, src_lengths,
                               adj_arc_in, adj_arc_out, adj_lab_in,
                               adj_lab_out, mask_in, mask_out,
                               mask_loop, mask_sent)

        else:
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)

        if self.context:
            dec_out, _, _ = self.model.decoder(
                tgt_in, memory_bank, dec_states, memory_lengths=src_lengths, context = context)
        else:
            dec_out, _, _ = self.model.decoder(
                tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
