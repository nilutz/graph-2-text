for num in 1 2 3 4 5 6 7 8 9 10
do
#train
python3 ../../train.py -data ../../data/football_delex_ctx_1 -save_model ../../data/football_delex_layer_${num}_ctx -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 25 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers ${num} -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_delex_layer_${num}_ctx -gcn_residual residual -seed ${num}

#translate
python3 ../../translate.py -model ../../data/football_delex_layer_${num}_ctx*e25.pt -data_type gcn -src ../../data/data-football/delex_/test-data-football-gcn-delex-src-nodes.txt -tgt ../../data/data-football/delex_/test-data-football-gcn-delex-tgt.txt -src_label ../../data/data-football/delex_/test-data-football-gcn-delex-src-labels.txt -src_node1 ../../data/data-football/delex_/test-data-football-gcn-delex-src-node1.txt -src_node2 ../../data/data-football/delex_/test-data-football-gcn-delex-src-node2.txt -src_ctx ../../data/data-football/delex_/test-data-football-gcn-delex-context.txt -output ../../data/data-football/delex_/delexicalized_predictions_test_layer_${num}.txt -context -replace_unk

#relex
python ../../football_processing/relex.py -t ../../data/data-football/delex_ -p delexicalized_predictions_test_layer_${num}.txt

#BLEU
sh ../../football_processing/calculate_bleu.sh ../../data/data-football/delex_/test-data-football-gcn-delex.reference  ../../data/data-football/delex_/relexicalised_predictions_layer_${num}.txt > out_layer_bleu_${num}
done