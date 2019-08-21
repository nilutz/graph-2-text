#WEBNLG
specific dependecies for webnlg project

	pip install -r requirements.txt

Download webnlg data from [here](https://gitlab.com/shimorina/webnlg-dataset/tree/master/webnlg_challenge_2017)
in data/webnlg/ keeping the three folders for the different partitions.


### WebNLG scripts ###

1. generate input files for GCN (note WebNLG dataset partitions 'train' and 'dev' are in *data/webnlg*
```
cd data/webnlg/
python3 webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ../data-webnlg/webnlg/
python3 webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ../data-webnlg/webnlg/ -p test -c seen #to process test partition
```
(Make sure the test directory only contains files from the WebNLG dataset, e.g., look out for .DS_Store files.)

If we want to have special arcs in the graph for multi-word named entities then add ```-e``` argument.
Otherwise the graph will contain a single node, e.g. The_Monument_To_War_Soldiers.


To make source and target tokens lowercased, add ```-l``` argument. This applies only to **notdelex** version.

2. relexicalise output of GCN
```
cd data/webnlg/
python3 webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ../data-webnlg/webnlg/ -f delexicalized_predictions_dev.txt
```
To relexicalise specific partition only, e.g. test add the following argument:
```-p test```

Note: The scripts now read the file 'delex_dict.json' from the same directory of main file (e.g. 'webnlg_gcnonmt_input.py')
Note: The sorting of the list of files is added but commented out
Note: the relexicalisation script should be run both for 'all-delex' and 'all-notdelex' too, as it does some other formattings needed before running the evaluation metrics.
```
python3 webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ../data-webnlg/webnlg/ -f delexicalized_predictions_dev.txt -c seen
```


3. metrics (generate files for METEOR and TER)
```
python3 webnlg_eval_scripts/metrics.py --td data/webnlg/ --pred data/webnlg/relexicalised_predictions.txt --p dev
```
