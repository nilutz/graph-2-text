#!/bin/bash

for file in ../../saved_runs/hyperparams/*.pt;do
 echo $file
 for ((i = 0; i < 3; i++))
    do
        name=${file##*/}
        base=${name%.pt}
        n=${base:62:15}
        NAME=delexicalized_predictions_hyper_${n}.txt
        SNAME=delexicalized_predictions_hyper_${n}
        PARAMS=hyper_${n}
    done
    MODELSRC=$file

    echo $MODELSRC
    echo $PARAMS

# # #train
# # # python3 ../../train.py -data ../../data/football_${type}_ctx_1 -save_model ../../data/football_${type}_${num}_${layer}_${con}_${embsize}_ctx -rnn_size ${embsize} -word_vec_size ${embsize} -layers 1 -epochs 25 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs ${embsize} -gcn_num_units ${embsize} -gcn_in_arcs -gcn_out_arcs -gcn_num_layers ${layer} -gcn_num_labels 5 -context -tensorboard -gpuid 0 -model_name football_${type}_${num}_${layer}_${con}_${embsize}_ctx -gcn_residual ${con} -seed ${num}

# # #translate
python ../../translate.py -model $MODELSRC -data_type gcn -src ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-src-nodes.txt -tgt ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-tgt.txt -src_label ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-src-labels.txt -src_node1 ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-src-node1.txt -src_node2 ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-src-node2.txt -src_ctx ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-context.txt -output ../../data/data-football/delex_attr_postfix_/$NAME -context -replace_unk

# #relex
python ../../football_processing/relex.py -t ../../data/data-football/delex_attr_postfix_ -p $NAME -r test-data-football-gcn-delex_attr_postfix.relex -out relexpredictions_$NAME

# #bleu
sh ../../football_processing/calculate_bleu.sh ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix.reference  ../../data/data-football/delex_attr_postfix_/relexpredictions_$NAME > out_bleu_delex_attr_postfix_$PARAMS.txt

# # #translate fake
python3 ../../translate.py -model $MODELSRC -data_type gcn -src ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-src-nodes.txt -tgt ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-tgt.txt -src_label ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-src-labels.txt -src_node1 ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-src-node1.txt -src_node2 ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-src-node2.txt -src_ctx ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-context.txt -output ../../data/data-football/delex_attr_postfix_/fake_$NAME -context -replace_unk

# # #relex fake
python3 ../../football_processing/relex.py -t ../../data/data-football/delex_attr_postfix_ -p fake_$NAME -r test-data-football-gcn-delex_attr_postfix.relex -out relexpredictions_fake_$NAME

# # conversion for ter and meteor
python ../../football_processing/metrics.py -t ../../data/data-football/delex_attr_postfix_/ -p relexpredictions_$NAME -r test-data-football-gcn-delex_attr_postfix.reference -partition $PARAMS

# # # #ctxe
python3 ../../football_processing/ctx_eval.py -p ../../data/data-football/delex_attr_postfix_/relexpredictions_$NAME -r ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix.reference -o ../../data/data-football/delex_attr_postfix_/ctx_eval_delex_attr_postfix_$PARAMS.txt -f ../../data/data-football/delex_attr_postfix_/relexpredictions_fake_$NAME -c ../../data/data-football/delex_attr_postfix_/test-data-football-gcn-delex_attr_postfix-context.txt -x ../../data/data-football/delex_attr_postfix_/test_fake-data-football-gcn-delex_attr_postfix-context.txt

# # # #TER produces files out_ter_{type}_{num}.txt with TER data
java -jar ../../eval_tools/tercom-master/tercom-0.10.0.jar -h ../../data/data-football/delex_attr_postfix_/relexpredictions_$SNAME-ter.txt -r ../../data/data-football/delex_attr_postfix_/hyperparams_1024_ctx-all-notdelex-refs-ter.txt > out_ter_delex_attr_postfix_$PARAMS.txt

# # # #METEOR
java -Xmx2G -jar ../../eval_tools/meteor-master/meteor-1.6.jar ../../data/data-football/delex_attr_postfix_/relexpredictions_$NAME ../../data/data-football/delex_attr_postfix_/hyperparams_1024_ctx-all-notdelex-refs-meteor.txt -r 8 -l de -norm > out_meteor_delex_attr_postfix_$PARAMS.txt

done