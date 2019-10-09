# produces METEOR and TER files from relexicalized_predictions

import plac
from pathlib import Path
import json

@plac.annotations(
    inputdir = ("dir to predfile", "option", "t", str),
    predfile = ('name of predfile', 'option', "p", str),
    refile = ('path to referencefile', 'option', 'r', str),
    partition = ('partition', 'option','partition', str)
)
def main(inputdir = '../data/data-football/delex_', partition = '', predfile = 'relexicalized_predictions_test.txt', refile = ''):
    
    references = []  # each element is a list of references
    pure_references = []
    initialref = inputdir + refile
    
    
    ## TER

    name = predfile.split('_')[-2] +'_'+predfile.split('_')[-1].replace('.txt','') 
    ##ter refs
    # complete refs with references for all sents
    with open(initialref, 'r') as f:
        for i, line in enumerate(f):
            references.append([line.strip() + ' (id' + str(i) + ')\n'])
            pure_references.append([line])

    # create a file with only one reference for TER
    with open(inputdir + partition + name +'-all-notdelex-oneref-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    with open(inputdir + partition + name +'-all-notdelex-refs-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    # ter preds
    with open( inputdir + predfile, 'r') as f:
        geners = [line.strip() + ' (id' + str(i) + ')\n' for i, line in enumerate(f)]
    #with open('relexicalised_predictions-ter.txt', 'w+') as f:
    with open(inputdir + predfile.replace('.txt','-ter.txt'), 'w+') as f:
        f.write(''.join(geners))


    #meteor refs

    # data for meteor
    # For N references, it is assumed that the reference file will be N times the length of the test file,
    # containing sets of N references in order.
    # For example, if N=4, reference lines 1-4 will correspond to test line 1, 5-8 to line 2, etc.
    with open(inputdir + partition + name + '-all-notdelex-refs-meteor.txt', 'w+') as f:
        for ref in pure_references:
            empty_lines = 8 - len(ref)  # calculate how many empty lines to add (8 max references)
            f.write(''.join(ref))
            if empty_lines > 0:
                f.write('\n' * empty_lines)
    print('Input files for METEOR and TER generated successfully.')


if __name__ == "__main__":
    plac.call(main)


#python ../../football_processing/metrics.py -t ../../data/data-football/delex_/ -p relexicalised_predictions_test_1.txt -r test-data-football-gcn-delex.reference          