# Foundations
This code is fork and is heavily based on the code used in the paper [Deep Graph Convolutional Encoders for Structured Data to Text Generation](http://aclweb.org/anthology/W18-6501) by Diego Marcheggiani and Laura Perez-Beltrachini. They leverage the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) library with a Graph Convolutional Network encoder.

Basically I added some code for the context vectors and my own dataset

### Dependencies
 - Python 3
 - [Pytorch 0.3.1](https://pytorch.org/get-started/locally/)
 - [Networkx](https://networkx.github.io) 
 - Docker

# Install
Clone and cd into directory. We use docker that installs all requirements in a docker container. Run:
	
	make build .

## Running on CPU

	make run 

## Running on GPU

	make run_gpu

# Reproducing the original Webnlg Experiment
read the 

	README-original.md

and run the commands

# Football Data Experiment

## Prepare data
Run [data-extractor]() and get a data_with_triples_full.pkl file and put it in data/data-football/, then for generating the input:
	
	cd football_preprocessing

	python preprocess.py -p ../data/data-football/sentences_full_head_lemma.pkl -o text -c -f

	python preprocess.py -p ../data/data-football/sentences_full_lower_head_lemma.pkl -o lower -c -f


	python preprocess.py -p ../data/data-football/sentences_full_lemma.pkl -o text -c -f

	python preprocess.py -p ../data/data-football/sentences_full_lower_lemma.pkl -o lower -c -f


## Preprocess
This generates the vocab and torchtext iterators

Exchange text_/delex_/delex_lower_/lower_ for different versions

	python preprocess.py -train_src data/football/text_/train-football-gcn-text-src-nodes.txt \
	-train_label data/football/text_/train-football-gcn-text-src-labels.txt \
	-train_node1 data/football/text_/train-football-gcn-text-src-node1.txt \
	-train_node2 data/football/text_/train-football-gcn-text-src-node2.txt \
	-train_tgt data/football/text_/train-football-gcn-text-tgt.txt \
	-valid_src data/football/text_/dev-football-gcn-text-src-nodes.txt \
	-valid_label data/football/text_/dev-football-gcn-text-src-labels.txt \
	-valid_node1 data/football/text_/dev-football-gcn-text-src-node1.txt \
	-valid_node2 data/football/text_/dev-football-gcn-text-src-node2.txt \
	-valid_tgt data/football/text_/dev-football-gcn-text-tgt.txt \
	-save_data data/gcn_exp_text_football -src_vocab_size 5000 -tgt_vocab_size 5000 -data_type gcn 


#### with context
	python preprocess.py -train_src data/football/text_/train-football-gcn-text-src-nodes.txt \
	-train_label data/football/text_/train-football-gcn-text-src-labels.txt \
	-train_node1 data/football/text_/train-football-gcn-text-src-node1.txt \
	-train_node2 data/football/text_/train-football-gcn-text-src-node2.txt \
	-train_tgt data/football/text_/train-football-gcn-text-tgt.txt \
	-valid_src data/football/text_/dev-football-gcn-text-src-nodes.txt \
	-valid_label data/football/text_/dev-football-gcn-text-src-labels.txt \
	-valid_node1 data/football/text_/dev-football-gcn-text-src-node1.txt \
	-valid_node2 data/football/text_/dev-football-gcn-text-src-node2.txt \
	-valid_tgt data/football/text_/dev-football-gcn-text-tgt.txt \
	-train_ctx data/football/text_/train-football-gcn-text-context.txt \
	-valid_ctx data/football/text_/dev-football-gcn-text-context.txt \
	-save_data data/gcn_exp_text_football_ctx -src_vocab_size 5000 -tgt_vocab_size 5000 -data_type gcn 

One can also run the original webnlg data with this repository just navigate to data/webnlg

## TRAIN
Training procedure:

	python3 train.py -data data/gcn_exp_text_football -save_model data/exp_football_ -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 10 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 1 -gcn_num_labels 5 -tensorboard -gpuid 0 -model_name text

#### with context
make sure the data has a context field

	python3 train.py -data data/gcn_exp_text_football_ctx -save_model data/1_exp_football_ctx_ -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 10 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 1 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name text_


## GENERATE
Generate with obtained model

	python3 translate.py -model data/exp_football__acc_55.09_ppl_12.18_e10.pt -data_type gcn -src data/football/text_/test-football-gcn-text-src-nodes.txt -tgt data/football/text_/test-football-gcn-text-tgt.txt -src_label data/football/text_/test-football-gcn-text-src-labels.txt -src_node1 data/football/text_/test-football-gcn-text-src-node1.txt -src_node2 data/football/text_/test-football-gcn-text-src-node2.txt -output data/football/delexicalized_predictions_test.txt -replace_unk -verbose -model_name text_


#### with context

	python3 translate.py -model data/1_exp_football_ctx__acc_52.40_ppl_15.87_e10.pt -data_type gcn -src data/football/text_/test-football-gcn-text-src-nodes.txt -tgt data/football/text_/test-football-gcn-text-tgt.txt -src_label data/football/text_/test-football-gcn-text-src-labels.txt -src_node1 data/football/text_/test-football-gcn-text-src-node1.txt -src_node2 data/football/text_/test-football-gcn-text-src-node2.txt -src_ctx data/football/text_/test-football-gcn-text-context.txt -output data/football/delexicalized_predictions_test_CONTEXT.txt -replace_unk -verbose -context -model_name text_

#### Also test with test_fake
we just flip the classes here:

	python3 translate.py -model data/1_exp_football_ctx__acc_52.40_ppl_15.87_e10.pt -data_type gcn -src data/football/text_/test_fake-football-gcn-text-src-nodes.txt -tgt data/football/text_/test_fake-football-gcn-text-tgt.txt -src_label data/football/text_/test_fake-football-gcn-text-src-labels.txt -src_node1 data/football/text_/test_fake-football-gcn-text-src-node1.txt -src_node2 data/football/text_/test_fake-football-gcn-text-src-node2.txt -src_ctx data/football/text_/test_fake-football-gcn-text-context.txt -output data/football/delexicalized_predictions_test_CONTEXT.txt -replace_unk -verbose -context -model_name text_

#### more params
add the following to get more statistic
	
	-report_bleu
	-report_rouge
	
## EVAL





* true:
SENT 1445: ('Brosinski', 'type', 'ASSIST', 'schicken', 'Hofmann', 'type', 'PLAYER', 'im', 'richtigen', 'Moment')
PRED 1445: Brosinski schickt Hofmann steil .
GOLD 1445: Brosinski schickt Hofmann , der genau im richtigen Moment startet , steil .
PRED SCORE: -14.5819
GOLD SCORE: -133.5462

* fake:
SENT 1445: ('Brosinski', 'type', 'ASSIST', 'schicken', 'Hofmann', 'type', 'PLAYER', 'im', 'richtigen', 'Moment')
PRED 1445: Brosinski schickte Hofmann , der im richtigen Moment steil schickte .
GOLD 1445: Brosinski schickt Hofmann , der genau im richtigen Moment startet , steil .
PRED SCORE: -21.8571
GOLD SCORE: -143.2870

# Ideas / Todo
* BPE
* make context into logging
* https://github.com/wouterkool/stochastic-beam-search
* https://github.com/huggingface/transfer-learning-conv-ai/blob/master/interact.py

# References

* [Controlling Linguistic Style Aspects in Neural Language Generation
](https://arxiv.org/abs/1707.02633)
* [Deep Graph Convolutional Encoders for Structured Data to Text Generation](http://aclweb.org/anthology/W18-6501) -> [code](https://github.com/diegma/graph-2-text)
* [ONMT-py](https://github.com/OpenNMT/OpenNMT-py/tree/master/onmt)