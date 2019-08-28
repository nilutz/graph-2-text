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
	
	cd football_processing

	python preprocess.py -c -f -p ../data/data-football/sentences_full_notdelex.pkl

	python preprocess.py -c -f -p ../data/data-football/sentences_full_delex.pkl


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


#### with context
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
	-train_ctx data/data-football/delex_/train-data-football-gcn-delex-context.txt \
	-valid_ctx data/data-football/delex_/dev-data-football-gcn-delex-context.txt \
	-save_data data/football_delex_ctx_1 -src_vocab_size 5005 -tgt_vocab_size 5005 -data_type gcn 

## TRAIN
Training procedure:

#### with context
make sure the data has a context field

	python3 train.py -data data/football_delex_ctx_1 -save_model data/football_delex_1_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_delex_1_ctx -gcn_residual residual -seed 43


## GENERATE
Generate with obtained model

#### with context

	python3 translate.py -model data/football_delex_1_ctx*e30.pt -data_type gcn -src data/data-football/delex_/test-data-football-gcn-delex-src-nodes.txt -tgt data/data-football/delex_/test-data-football-gcn-delex-tgt.txt -src_label data/data-football/delex_/test-data-football-gcn-delex-src-labels.txt -src_node1 data/data-football/delex_/test-data-football-gcn-delex-src-node1.txt -src_node2 data/data-football/delex_/test-data-football-gcn-delex-src-node2.txt -src_ctx data/data-football/delex_/test-data-football-gcn-delex-context.txt -output data/data-football/delex_/delexicalized_predictions_test1.txt -context -replace_unk -verbose -report_bleu 

#### Also test with test_fake
we just flip the classes here:

	

#### more params
add the following to get more statistic
	
	-report_bleu
	-report_rouge
	
# Postprocessing
if you've trained with *delex* you need to relex the Entity Desriptions to effectivly compare to the reference

	python relex.py 

creates relexicalised_predictions_test.txt

# EVAL
to compare go into folder

	cd data/data-football/delex_

	../../../football_processing/calculate_bleu.sh

and there you have the BLEU score.

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