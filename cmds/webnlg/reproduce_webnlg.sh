#!/bin/bash

for num in 1 2 3
do
#train
python3 ../../train.py -data ../../data/webnlg_delex_1 -save_model ../../data/webnlg_delex_${num}_notembd -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 30 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 4 -gcn_num_labels 5 -tensorboard -gpuid 0 -model_name webnlg_delex_${num}_notembd -gcn_residual residual -seed ${num}

#translate
python3 ../../translate.py -model ../../data/webnlg_delex_${num}_notembd*e30.pt -data_type gcn -src ../../data/data-webnlg/test-webnlg-all-delex-src-nodes.txt -tgt ../../data/data-webnlg/test-webnlg-all-delex-tgt.txt -src_label ../../data/data-webnlg/test-webnlg-all-delex-src-labels.txt -src_node1 ../../data/data-webnlg/test-webnlg-all-delex-src-node1.txt -src_node2 ../../data/data-webnlg/test-webnlg-all-delex-src-node2.txt -output ../../data/data-webnlg/delexicalized_predictions_test_${num}.txt -replace_unk


#relex
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ../../data/data-webnlg/webnlg/ -f ../../data/data-webnlg/delexicalized_predictions_test_${num}.txt -c seen -p test -g _${num}

#Bleu
../../webnlg_eval_scripts/calculate_bleu_dev_input.sh relexicalised_predictions_${num}.txt


#scripts for meteor and ter files
python3 ../../webnlg_eval_scripts/metrics.py --td ../../data/data-webnlg/ --pred relexicalised_predictions_${num}.txt --p test

#TER
java -jar ../../eval_tools/tercom-master/tercom-0.10.0.jar -r ../../data/data-webnlg/relexicalised_predictions_${num}-ter.txt -h ../../data/data-webnlg/test-all-notdelex-refs-ter.txt > out_${num}.txt

done

#calculate Bleu and AVG with delexicalized_predictions_test_${num}.txt