#!/bin/bash

for type in delex notdelex
do
for num in 1 2 3
do
#train

python3 ../../train.py -data ../../data/football_${type}_ctx_1 -save_model ../../data/football_${type}_${num}_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_${type}_${num}_ctx -gcn_residual residual -seed ${num}

#translate
python3 ../../translate.py -model ../../data/football_${type}_${num}_ctx*e30.pt -data_type gcn -src ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-nodes.txt -tgt ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -src_label ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-labels.txt -src_node1 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node1.txt -src_node2 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node2.txt -src_ctx ../../data/data-football/${type}_/test-data-football-gcn-${type}-context.txt -output ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt -context -replace_unk
done
done
#calculate Bleu and AVG with delexicalized_predictions_test_${num}.txt