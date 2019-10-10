This directory contains convenience shell scrips that runs whole experiments and puts results into the respective folders.

cd into football or webnlg and start the sh scripts.


# Football

    cd football

## Experiment 1

    sh delex_attr_1.sh
    sh delex_attr_postfix_1.sh
    sh notdelex_attr_1.sh
    sh notdelex_attr_postfix_1.sh

    python statistics.py -e 1

## Experiment 2

    sh experiment2.sh
    python statistics.py -e 2

## Experiment 3

    sh experiment3.sh
    python statistics.py -e 3


# Webnlg

    cd webnlg

    reproduce_webnlg.sh
    python statistics.py