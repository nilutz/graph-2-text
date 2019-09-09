#!/bin/bash

for type in delex notdelex
do

for num in 1 2 3
do
#train

python3 ../../train.py -data ../../data/football_${type}_ctx_1 -save_model ../../data/football_${type}_${num}_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_${type}_${num}_ctx -gcn_residual residual -seed ${num}

#translate
python3 ../../translate.py -model ../../data/football_${type}_${num}_ctx*e30.pt -data_type gcn -src ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-nodes.txt -tgt ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -src_label ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-labels.txt -src_node1 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node1.txt -src_node2 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node2.txt -src_ctx ../../data/data-football/${type}_/test-data-football-gcn-${type}-context.txt -output ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt -context -replace_unk

#relex
python ../../football_processing/relex.py -t ../../data/data-football/${type}_ -p delexicalized_predictions_test_${num}.txt

#BLEU
#sh ../../football_processing/calculate_bleu.sh ../../data/data-football/${type}_/test-data-football-gcn-${type}.reference  ../../data/data-football/${type}_/relexicalised_predictions_test_${num}.txt 

if [${type} == delex]
then #delex

#make fake files
#python3 ../../translate.py -model ../../data/football_delex_1_ctx*e30.pt -data_type gcn -src ../../data/data-football/delex_/test_fake-data-football-gcn-delex-src-nodes.txt -tgt ../../data/data-football/delex_/test_fake-data-football-gcn-delex-tgt.txt -src_label ../../data/data-football/delex_/test_fake-data-football-gcn-delex-src-labels.txt -src_node1 ../../data/data-football/delex_/test_fake-data-football-gcn-delex-src-node1.txt -src_node2 ../../data/data-football/delex_/test_fake-data-football-gcn-delex-src-node2.txt -src_ctx ../../data/data-football/delex_/test_fake-data-football-gcn-delex-context.txt -output ../../data/data-football/delex_/delexicalized_predictions_test_fake_1.txt -context -replace_unk

sh ../../football_processing/calculate_bleu.sh ../../data/data-football/delex_/test-data-football-gcn-delex.reference  ../../data/data-football/delex_/relexicalised_predictions_test_${num}.txt 

#metrics
python ../../football_processing/metrics.py -t ../../data/data-football/${type}_/ -p relexicalised_predictions_test_${num}.txt -r test-data-football-gcn-delex.reference


else #notedelex
sh ../../football_processing/calculate_bleu.sh ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt

#metrics
python ../../football_processing/metrics.py -t ../../data/data-football/${type}_/ -p relexicalised_predictions_test_${num}.txt -r test-data-football-gcn-${type}-tgt.txt


fi

#TER produces files out_{num}.txt with TER data
java -jar ../../eval_tools/tercom-master/tercom-0.10.0.jar -h ../../data/data-football/${type}/relexicalised_predictions_test_${num}-ter.txt -r ../../data/data-football/${type}/test_${num}-all-notdelex-refs-ter.txt > out_ter_${type}_${num}.txt

#METEOR
java -Xmx2G -jar ../../eval_tools/meteor-master/meteor-1.6.jar ../../data/data-football/${type}/relexicalised_predictions_test_${num}.txt ../../data/data-football/${type}/test_${num}-all-notdelex-refs-meteor.txt -r 8 -l de -norm

done
done