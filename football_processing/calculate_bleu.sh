#!/bin/bash

# compute BLEU

export TEST_TARGETS_REF0=test-data-football-gcn-delex.reference
export TEST_TARGETS_PREDS=relexicalised_predictions_test.txt

../../../webnlg_eval_scripts/multi-bleu.perl ${TEST_TARGETS_REF0} < ${TEST_TARGETS_PREDS}
