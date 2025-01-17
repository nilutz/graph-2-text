This is the code used in the paper [Deep Graph Convolutional Encoders for Structured Data to Text Generation](http://aclweb.org/anthology/W18-6501) by Diego Marcheggiani and Laura Perez-Beltrachini.

We extended the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) library with a Graph Convolutional Network encoder.



### Dependencies
 - Python 3
 - [Pytorch 0.3.1](https://pytorch.org/get-started/locally/)
 - [Networkx](https://networkx.github.io) 


### Download and prepare data


Download webnlg data from [here](https://gitlab.com/shimorina/webnlg-dataset/tree/master/webnlg_challenge_2017)
in data/webnlg/ keeping the three folders for the different partitions.

There is a preparation step for extracting the node and the edges from the graphs.
Instruction for this are in the *WebNLG Scripts* section point 1, at the bottom of the readme.

The preprocessing training and generation steps are the same for the Surface Realization Task (SR11) data.
The SR11 data can be downloaded from [here](https://sites.google.com/site/genchalrepository/surface-realisation/sr11).
Data preparation scripts are on the *SR11 Scripts* section.

### Preprocess

! file paths are changed(compared to the original code) to fit the structure of this fork !

Using the files obtained in the preparation step, we first generate data and dictionary for OpenNmt.


To preprocess the raw files run:

```
python3 preprocess.py -train_src data/data-webnlg/train-webnlg-all-notdelex-src-nodes.txt \
-train_label data/data-webnlg/train-webnlg-all-notdelex-src-labels.txt \
-train_node1 data/data-webnlg/train-webnlg-all-notdelex-src-node1.txt \
-train_node2 data/data-webnlg/train-webnlg-all-notdelex-src-node2.txt \
-train_tgt data/data-webnlg/train-webnlg-all-notdelex-tgt.txt \
-valid_src data/data-webnlg/dev-webnlg-all-notdelex-src-nodes.txt \
-valid_label data/data-webnlg/dev-webnlg-all-notdelex-src-labels.txt \
-valid_node1 data/data-webnlg/dev-webnlg-all-notdelex-src-node1.txt \
-valid_node2 data/data-webnlg/dev-webnlg-all-notdelex-src-node2.txt \
-valid_tgt data/data-webnlg/dev-webnlg-all-notdelex-tgt.txt \
-save_data data/webnlg_1_notdelex -src_vocab_size 5000 -tgt_vocab_size 5000 -data_type gcn  
```

The argument ```-dynamic_dict``` is needed to train models using copy mechanism e.g., the model GCN_CE in the paper.

Preprocessing step for the SR11 task is the same as WebNLG.

#### Embeddings

Using pre-trained embeddings in OpenNMT, need to do this pre-processing step first:
```
export glove_dir="../vectors"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/gcn_exp_webnlg1_delex.vocab.pt" \
    -output_file "data/gcn_exp.embeddings" 
```

### Train
After you preprocessed the files you can run the training procedure:
```

python3 train.py -data data/gcn_exp_webnlg1_delex -save_model data/webnlg1_delex_emb -rnn_size 256 -word_vec_size 256 -layers 1 -epochs 10 -optim adam -learning_rate 0.001 -encoder_type gcn -gcn_num_inputs 256 -gcn_num_units 256 -gcn_in_arcs -gcn_out_arcs -gcn_num_layers 1 -gcn_num_labels 5 -tensorboard -gpuid 0 -model_name webnlg1_delex_emb 
```

To train with a GCN encoder the following options must be set:
<code><pre>
-encoder_type  
-gcn_num_inputs Input size for the gcn layer
-gcn_num_units Output size for the gcn layer
-gcn_num_labels Number of labels for the edges of the gcn layer
-gcn_num_layers Number of gcn layers
-gcn_in_arcs Use incoming edges of the gcn layer
-gcn_out_arcs Use outgoing edges of the gcn layer
-gcn_residual Decide wich skip connection to use between GCN layers 'residual' or 'dense' default it is set as no resiudal connections
-gcn_use_gates  Switch to activate edgewise gates
-gcn_use_glus Node gates
</code></pre>


Add the following arguments to use pre-trained embeddings:
```
        -pre_word_vecs_enc data/gcn_exp.embeddings.enc.pt \
        -pre_word_vecs_dec data/gcn_exp.embeddings.dec.pt \
```

### Generate ###
Generating with obtained model:
DEV SET
```
python3 translate.py -model data/webnlg3Emb__webnlg3Emb_acc_70.35_ppl_3.29_e10.pt -data_type gcn -src data/data-webnlg/dev-webnlg-all-notdelex-src-nodes.txt -tgt data/data-webnlg/dev-webnlg-all-notdelex-tgt.txt -src_label data/data-webnlg/dev-webnlg-all-notdelex-src-labels.txt -src_node1 data/data-webnlg/dev-webnlg-all-notdelex-src-node1.txt -src_node2 data/data-webnlg/dev-webnlg-all-notdelex-src-node2.txt -output data/data-webnlg/delexicalized_predictions_dev.txt -replace_unk -verbose -report_bleu
```

TEST SET
```
python3 translate.py -model data/webnlg3_delex_emb_webnlg3_delex_emb_acc_65.82_ppl_4.18_e17.pt -data_type gcn -src data/data-webnlg/test-webnlg-all-delex-src-nodes.txt -tgt data/data-webnlg/test-webnlg-all-delex-tgt.txt -src_label data/data-webnlg/test-webnlg-all-delex-src-labels.txt -src_node1 data/data-webnlg/test-webnlg-all-delex-src-node1.txt -src_node2 data/data-webnlg/test-webnlg-all-delex-src-node2.txt -output data/data-webnlg/delexicalized_predictions_test.txt -replace_unk -verbose -report_bleu
```

### Postprocessing and Evaluation ###
For post processing follow step 2 and 3 of WebNLG scripts.
For evaluation follow the instruction of the WebNLG challenge [baseline](http://webnlg.loria.fr/pages/baseline.html) or run 

    cd data/data-webnlg/
    ../../webnlg_eval_scripts/calculate_bleu_dev.sh 
For the SR11 task, scripts for the 3 metrics are the same as used for WebNLG [see](https://www.aclweb.org/anthology/W11-2832).

### WebNLG scripts ###

1. generate input files for GCN (note WebNLG dataset partitions 'train' and 'dev' are in *graph2text/webnlg-baseline/data/webnlg/*
```
cd data/data-webnlg/
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./webnlg/
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./webnlg/ -p test -c seen #to process test partition
```
(Make sure the test directory only contains files from the WebNLG dataset, e.g., look out for .DS_Store files.)


If we want to have special arcs in the graph for multi-word named entities then add ```-e``` argument.
Otherwise the graph will contain a single node, e.g. The_Monument_To_War_Soldiers.

```
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./webnlg/ -e
```

To make source and target tokens lowercased, add ```-l``` argument. This applies only to **notdelex** version.

2. relexicalise output of GCN
```
cd data/data-webnlg/
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ./webnlg/ -f ../data-webnlg/delexicalized_predictions_dev.txt
```
To relexicalise specific partition only, e.g. test add the following argument:
```-p test```

Note: The scripts now read the file 'delex_dict.json' from the same directory of main file (e.g. 'webnlg_gcnonmt_input.py')
Note: The sorting of the list of files is added but commented out
Note: the relexicalisation script should be run both for 'all-delex' and 'all-notdelex' too, as it does some other formattings needed before running the evaluation metrics.
```
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ./webnlg/ -f ../data-webnlg/delexicalized_predictions_test.txt -c seen
```


3. metrics (generate files for METEOR and TER)
```
python3 webnlg_eval_scripts/metrics.py --td data/data-webnlg/ --pred data/data-webnlg/relexicalised_predictions.txt --p dev
```

### SR11 scripts ###

Generate/format input dataset for gcn encoder:
```
cd srtask/
python3 sr11_onmtgcn_input.py -i ../data/srtask11/SR_release1.0/ -t deep
python3 sr11_onmtgcn_input.py -i ../data/srtask11/SR_release1.0/ -t deep -p test
```
reanonymise:
```
python3 sr_onmtgcn_deanonymise.py -i ../data/srtask11/SR_release1.0/ -f ../data/srtask11/SR_release1.0/devel-sr11-deep-anonym-tgt.txt -p devel -t deep
```
generate/format input dataset for linearised input and sequence encoder:
```
cd srtask/
python3 sr11_linear_input.py -i ../data/srtask11/SR_release1.0/ -t deep
python3 sr11_linear_input.py -i ../data/srtask11/SR_release1.0/ -t deep -p test
```

generate TER input files
```
python3 srtask/srpredictions4ter.py --pred PREDSFILE --gold data/srtask11/SR_release1.0/test/SRTESTB_sents.txt
```
```PREDSFILE``` is filename with relative path


### Citation
```
@inproceedings{marcheggiani-perez-beltrachini-2018-deep,
    title = "Deep Graph Convolutional Encoders for Structured Data to Text Generation",
    author = "Marcheggiani, Diego  and Perez-Beltrachini, Laura",
    booktitle = "Proceedings of the 11th International Conference on Natural Language Generation",
    month = nov,
    year = "2018",
    address = "Tilburg University, The Netherlands",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6501",
    pages = "1--9"
}
```
