#!/bin/bash

# compute BLEU

export TEST_TARGETS_REF0=../../data/test-data-football-gcn-delex.reference
export TEST_TARGETS_PREDS=relexicalised_predictions_test.txt

../../webnlg_eval_scripts/multi-bleu.perl $1 < $2
