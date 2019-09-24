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

# Reproducing the original WebNLG Task
read the 

	README-original.md

and run the commands or go to cmds/webnlg to run an sh script running all in order

# Football Task

## Prepare data
Run [data-extractor]() and get a sentences_full_{not}delex.pkl file and put it in data/data-football/, then for generating the input:
	
	cd football_processing

	#python preprocess.py -c -f -p ../data/data-football/sentences_full_notdelex.pkl

	python preprocess.py -c -f -p ../data/data-football/sentences_full_delex_attr_postfix.pkl


## Preprocess
This generates the vocab and torchtext iterators

Exchange delex_/not_delex_ for different versions

	python preprocess.py -train_src data/data-football/delex_/train-data-football-gcn-delex-src-nodes.txt \
	-train_label data/data-football/delex_/train-data-football-gcn-delex-src-labels.txt \
	-train_node1 data/data-football/delex_/train-data-football-gcn-delex-src-node1.txt \
	-train_node2 data/data-football/delex_/train-data-football-gcn-delex-src-node2.txt \
	-train_tgt data/data-football/delex_/train-data-football-gcn-delex-tgt.txt \
	-valid_src data/data-football/delex_/dev-data-football-gcn-delex-src-nodes.txt \
	-valid_label data/data-football/delex_/dev-data-football-gcn-delex-src-labels.txt \
	-valid_node1 data/data-football/delex_/dev-data-football-gcn-delex-src-node1.txt \
	-valid_node2 data/data-football/delex_/dev-data-football-gcn-delex-src-node2.txt \
	-valid_tgt data/data-football/delex_/dev-data-football-gcn-delex-tgt.txt \
	-save_data data/football_delex_1 -src_vocab_size 5000 -tgt_vocab_size 5000 -data_type gcn 


#### with context delex
	python preprocess.py -train_src data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-src-nodes.txt \
	-train_label data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-src-labels.txt \
	-train_node1 data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-src-node1.txt \
	-train_node2 data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-src-node2.txt \
	-train_tgt data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-tgt.txt \
	-valid_src data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-src-nodes.txt \
	-valid_label data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-src-labels.txt \
	-valid_node1 data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-src-node1.txt \
	-valid_node2 data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-src-node2.txt \
	-valid_tgt data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-tgt.txt \
	-train_ctx data/data-football/delex_attr_postfix_/train-data-football-gcn-delex_attr_postfix-context.txt \
	-valid_ctx data/data-football/delex_attr_postfix_/dev-data-football-gcn-delex_attr_postfix-context.txt \
	-save_data data/football_delex_attr_postfix_ctx_1 -src_vocab_size 5005 -tgt_vocab_size 5005 -data_type gcn  

#### with context notdelex
	python preprocess.py -train_src data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-src-nodes.txt \
	-train_label data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-src-labels.txt \
	-train_node1 data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-src-node1.txt \
	-train_node2 data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-src-node2.txt \
	-train_tgt data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-tgt.txt \
	-valid_src data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-src-nodes.txt \
	-valid_label data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-src-labels.txt \
	-valid_node1 data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-src-node1.txt \
	-valid_node2 data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-src-node2.txt \
	-valid_tgt data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-tgt.txt \
	-train_ctx data/data-football/notdelex_attr_postfix_/train-data-football-gcn-notdelex_attr_postfix-context.txt \
	-valid_ctx data/data-football/notdelex_attr_postfix_/dev-data-football-gcn-notdelex_attr_postfix-context.txt \
	-save_data data/football_notdelex_attr_postfix_ctx_1 -src_vocab_size 5005 -tgt_vocab_size 5005 -data_type gcn 

	
## TRAIN
Training procedure:

#### with context
make sure the data has a context field

	python3 train.py -data data/football_{type}_ctx_1 -save_model data/football_delex_1_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_delex_1_ctx -gcn_residual residual -seed 43


## GENERATE
Generate with obtained model

#### with context

	python3 translate.py -model data/football_delex_1_ctx*e30.pt -data_type gcn -src data/data-football/delex_/test-data-football-gcn-delex-src-nodes.txt -tgt data/data-football/delex_/test-data-football-gcn-delex-tgt.txt -src_label data/data-football/delex_/test-data-football-gcn-delex-src-labels.txt -src_node1 data/data-football/delex_/test-data-football-gcn-delex-src-node1.txt -src_node2 data/data-football/delex_/test-data-football-gcn-delex-src-node2.txt -src_ctx data/data-football/delex_/test-data-football-gcn-delex-context.txt -output data/data-football/delex_/delexicalized_predictions_test1.txt -context -replace_unk -verbose -report_bleu 

 => Also test with test_fake
	
## Postprocessing
if you've trained with *delex* you need to relex the Entity Desriptions to effectivly compare to the reference

	python ../../football_processing/relex.py -t ../../data/data-football/${type}_ -p delexicalized_predictions_test_${num}.txt

creates relexicalised_predictions_test.txt

## EVAL
For evaluation cd in football_preprocessing and run a preprocessing script.

	python metrics.py -t ../data/data-football/${type}_/ -p relexicalised_predictions_test_${num}.txt -r test-data-football-gcn-delex.reference


### BLEU
	sh football_processing/calculate_bleu.sh ../data/data-football/delex_/test-data-football-gcn-delex.reference  ../data/data-football/delex_/relexicalised_predictions_test_${num}.txt > out_bleu_${type}_${num}.txt


### METEOR
	java -Xmx2G -jar ../../eval_tools/meteor-master/meteor-1.6.jar ../../data/data-football/${type}/relexicalised_predictions_test_${num}.txt ../../data/data-football/${type}/test_${num}-all-notdelex-refs-meteor.txt -r 8 -l de -norm

### TER
	java -jar ../../eval_tools/tercom-master/tercom-0.10.0.jar -h ../../data/data-football/${type}/relexicalised_predictions_test_${num}-ter.txt -r ../../data/data-football/${type}/test_${num}-all-notdelex-refs-ter.txt > out_ter_${type}_${num}.txt

### CTXE
	python3 ../../football_processing/ctx_eval.py -p ../../data/data-football/${type}_/relexicalised_predictions_fake_${num}.txt -r ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -o ../../data/data-football/${type}/ctx_eval_${num}.txt -f ../../data/data-football/${type}_/relexicalised_predictions_fake_${num}.txt -c ../../data/data-football/${type}_/test-data-football-gcn-delex-context.txt -x ../../data/data-football/${type}_/test_fake-data-football-gcn-delex-context.txt


# CMDS
the /cmds folder provides sh script running the above command in order with correct type and numbering, for both webnlg and football tasks.



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
* [METEOR](https://www.cs.cmu.edu/~alavie/METEOR/README.html)
* [TER](https://github.com/jhclark/tercom)