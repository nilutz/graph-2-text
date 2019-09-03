# What we know from the paper


### baseline
-  The baseline model for the WebNLG task uses one layer
bidirectional LSTM encoder and one layer LSTM
decoder with embeddings and hidden units set to
256 dimensions .

### gcn
-  embedding 256 -> no glove?
- 4 gcn layers
- Residual connections
- Gates ?

Use delex for webnlg
- AVG of 3

Webnlg 18102. test seen(971)

gcn_ec:
need to setup with dynamic_dict params
- uses glove
- Copy mechanism
- 6 gcn layers
- Hidden 300

# EXP1 - delex normal gcn

## Preprocessing

    python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./webnlg/
    python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./webnlg/ -p test -c seen

    ##test
        wc -l test-webnlg-all-delex-tgt.txt
        2494 test-webnlg-all-delex-tgt.txt

        sort test-webnlg-all-delex.triple | uniq | wc -l
        971

        wc -l dev-webnlg-all-delex-tgt.txt
        2267 dev-webnlg-all-delex-tgt.txt
        
        sort dev-webnlg-all-delex.triple | uniq | wc -l
        871

        wc -l train-webnlg-all-delex-tgt.txt
        18101 train-webnlg-all-delex-tgt.txt

        => fits the paper...

    python3 preprocess.py -train_src data/data-webnlg/train-webnlg-all-delex-src-nodes.txt -train_label data/data-webnlg/train-webnlg-all-delex-src-labels.txt -train_node1 data/data-webnlg/train-webnlg-all-delex-src-node1.txt -train_node2 data/data-webnlg/train-webnlg-all-delex-src-node2.txt -train_tgt data/data-webnlg/train-webnlg-all-delex-tgt.txt -valid_src data/data-webnlg/dev-webnlg-all-delex-src-nodes.txt -valid_label data/data-webnlg/dev-webnlg-all-delex-src-labels.txt -valid_node1 data/data-webnlg/dev-webnlg-all-delex-src-node1.txt -valid_node2 data/data-webnlg/dev-webnlg-all-delex-src-node2.txt -valid_tgt data/data-webnlg/dev-webnlg-all-delex-tgt.txt -save_data data/webnlg_delex_1 -src_vocab_size 5000 -tgt_vocab_size 5000 -data_type gcn


    ## train
    webnlg_delex_{1,3}_notembd 
    seed {42,43,44}

        python3 train.py -data data/webnlg_delex_1 -save_model data/webnlg_delex_2_notembd -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -tensorboard -gpuid 0 -model_name webnlg_delex_2_notembd -gcn_residual residual -seed 43

        or with ?
        -gcn_use_gates

    
    ## translate with test

        python3 translate.py -model data/webnlg_delex_2_notembd*e30.pt -data_type gcn -src data/data-webnlg/test-webnlg-all-delex-src-nodes.txt -tgt data/data-webnlg/test-webnlg-all-delex-tgt.txt -src_label data/data-webnlg/test-webnlg-all-delex-src-labels.txt -src_node1 data/data-webnlg/test-webnlg-all-delex-src-node1.txt -src_node2 data/data-webnlg/test-webnlg-all-delex-src-node2.txt -output data/data-webnlg/delexicalized_predictions_test.txt -replace_unk -verbose -report_bleu 

    ## results

        cd data/data-webnlg/
        python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ./webnlg/ -f ../data-webnlg/delexicalized_predictions_test_1.txt -c seen -p test -gpuid 1

        ../../webnlg_eval_scripts/calculate_bleu_dev.sh #!look it the script make sure you compare to test not dev



=> this works as epected now:

# RESULTS

BLEU = 52.57, 85.3/64.8/50.7/40.4 (BP=0.906, ratio=0.910, hyp_len=19266, ref_len=21168)
BLEU = 52.71, 84.7/64.3/50.2/40.0 (BP=0.916, ratio=0.920, hyp_len=19289, ref_len=20972)
BLEU = 52.72, 84.4/64.2/49.9/39.4 (BP=0.923, ratio=0.926, hyp_len=19383, ref_len=20939)



# TRY gates

python3 train.py -data data/webnlg_delex_1 -save_model data/webnlg_delex_4_notembd_gates -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -tensorboard -gpuid 0 -model_name webnlg_delex_4_notembd_gates -gcn_residual residual -seed 44 -gcn_use_gates

python3 translate.py -model data/webnlg_delex_4_notembd_gates*e30.pt -data_type gcn -src data/data-webnlg/test-webnlg-all-delex-src-nodes.txt -tgt data/data-webnlg/test-webnlg-all-delex-tgt.txt -src_label data/data-webnlg/test-webnlg-all-delex-src-labels.txt -src_node1 data/data-webnlg/test-webnlg-all-delex-src-node1.txt -src_node2 data/data-webnlg/test-webnlg-all-delex-src-node2.txt -output data/data-webnlg/delexicalized_predictions_test.txt -replace_unk -verbose -report_bleu 


BLEU = 50.72, 82.4/61.7/47.8/37.9 (BP=0.921, ratio=0.924, hyp_len=19972, ref_len=21618)


# more metrics

## TER

python3 webnlg_eval_scripts/metrics.py --td data/data-webnlg/ --pred data/data-webnlg/relexicalised_predictions.txt --p test

java -jar eval_tools/tercom-master/tercom-0.10.0.jar -r data/data-webnlg/relexicalised_predictions-ter.txt -h test-all-notdelex-refs-ter.txt > out.txt