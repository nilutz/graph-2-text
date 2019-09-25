#!/bin/bash

for type in notdelex_attr_postfix
do
for num in 1
do

#train
python3 ../../train.py -data ../../data/football_${type}_ctx_1 -save_model ../../data/football_${type}_${num}_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_${type}_${num}_ctx -gcn_residual residual -seed ${num}

#translate
python3 ../../translate.py -model ../../data/football_${type}_${num}_ctx*e30.pt -data_type gcn -src ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-nodes.txt -tgt ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -src_label ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-labels.txt -src_node1 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node1.txt -src_node2 ../../data/data-football/${type}_/test-data-football-gcn-${type}-src-node2.txt -src_ctx ../../data/data-football/${type}_/test-data-football-gcn-${type}-context.txt -output ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt -context -replace_unk

#relex
#python ../../football_processing/relex.py -t ../../data/data-football/${type}_ -p delexicalized_predictions_test_${num}.txt -r test-data-football-gcn-${type}.relex

#translate fake
python3 ../../translate.py -model ../../data/football_${type}_${num}_ctx*e30.pt -data_type gcn -src ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-src-nodes.txt -tgt ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-tgt.txt -src_label ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-src-labels.txt -src_node1 ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-src-node1.txt -src_node2 ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-src-node2.txt -src_ctx ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-context.txt -output ../../data/data-football/${type}_/delexicalized_predictions_test_fake_${num}.txt -context -replace_unk

#relex fake
#python3 ../../football_processing/relex.py -t ../../data/data-football/${type}_ -p delexicalized_predictions_test_fake_${num}.txt -r test-data-football-gcn-${type}.relex

#bleu
sh ../../football_processing/calculate_bleu.sh ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt  ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt > out_bleu_${type}_${num}.txt

#conversion for ter and meteor
python ../../football_processing/metrics.py -t ../../data/data-football/${type}_/ -p delexicalized_predictions_test_${num}.txt -r test-data-football-gcn-${type}-tgt.txt

#ctxe
python3 ../../football_processing/ctx_eval.py -p ../../data/data-football/${type}_/delexicalized_predictions_fake_${num}.txt -r ../../data/data-football/${type}_/test-data-football-gcn-${type}-tgt.txt -o ../../data/data-football/${type}_/ctx_eval_${num}.txt -f ../../data/data-football/${type}_/relexicalised_predictions_fake_${num}.txt -c ../../data/data-football/${type}_/test-data-football-gcn-${type}-context.txt -x ../../data/data-football/${type}_/test_fake-data-football-gcn-${type}-context.txt

#TER produces files out_ter_{type}_{num}.txt with TER data
java -jar ../../eval_tools/tercom-master/tercom-0.10.0.jar -h ../../data/data-football/${type}_/delexicalized_predictions_test_${num}-ter.txt -r ../../data/data-football/${type}_/test_${num}-all-notdelex-refs-ter.txt > out_ter_${type}_${num}.txt

#METEOR
java -Xmx2G -jar ../../eval_tools/meteor-master/meteor-1.6.jar ../../data/data-football/${type}_/delexicalized_predictions_test_${num}.txt ../../data/data-football/${type}_/test_${num}-all-notdelex-refs-meteor.txt -r 8 -l de -norm > out_meteor_${type}_${num}.txt


done
done