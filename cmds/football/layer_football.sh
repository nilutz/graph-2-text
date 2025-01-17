#!/bin/bash

for type in delex_attr_postfix delex_attr_postfix_simple
do
for num in 103
do
for layer in 2 3 5 7 9 
do
for con in dense residual
do
for embsize in 256 512 1024
do

#train
python3 ../../train.py -data ../../data/football_${type}_ctx_1 -save_model ../../data/football_${type}_${num}_${layer}_${con}_${embsize}_ctx -rnn_size ${embsize} -word_vec_size ${embsize} -layers 1 -epochs 25 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs ${embsize} -gcn_num_units ${embsize} -gcn_in_arcs -gcn_out_arcs -gcn_num_layers ${layer} -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_${type}_${num}_${layer}_${con}_${embsize}_ctx -gcn_residual ${con} -seed ${num}

#translate
python3 ../../translate.py -model ../../data/football_${type}_${num}_${layer}_${con}_${embsize}_ctx*e25.pt -data_type gcn -src ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-nodes.txt -tgt ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -src_label ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-labels.txt -src_node1 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node1.txt -src_node2 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node2.txt -src_ctx ../../data/data-football/${type}_/test-data-football-gcn-${type}-context.txt -output ../../data/data-football/${type}_/delexicalized_predictions_test_${num}_${layer}_${con}_${embsize}_ctx.txt -context -replace_unk

#relex
python ../../football_processing/relex.py -t ../../data/data-football/${type}_ -p delexicalized_predictions_test_${num}_${layer}_${con}_${embsize}_ctx.txt -r test-data-football-gcn-${type}.relex

#bleu
#sh ../../football_processing/calculate_bleu.sh ../../data/data-football/${type}_/test-data-football-gcn-${type}.reference  ../../data/data-football/${type}_/relexicalised_predictions_test_${num}.txt > out_bleu_${type}_${num}_layer_${layer}.txt
done
done
done
done
done