import plac
from pathlib import Path
import json

@plac.annotations(
    topdir = ("dir to predfile", "option", "t", Path),
    predfile = ('name of predfile', 'option', "p", Path),
)
def main(topdir = '../data/data-football/delex_', predfile = 'delexicalized_predictions_test.txt', out = 'relex_predictions_test.txt'):

    print(topdir / predfile)
    with open( str(topdir / predfile) , 'r') as f:
        predictions = [line for line in f]

    parts = predfile.parts[0].split('_')

    with open( str(topdir / Path(parts[-1].split('.')[0]+'-data-football-gcn-delex.relex')), 'r') as f:
        relex = [json.loads(line) for line in f]
    
    relex_sents = []
    
    for pred, rel in zip(predictions, relex):
        
        relex_sentence = []
        for token in pred.split(' '):
            if token.isupper() and token in rel.keys():
                relex_sentence.append(rel[token][0])
            else:
                relex_sentence.append(token)
        
        relex_sents.append(' '.join(relex_sentence))

    outfileName =   'relexicalised_predictions_'+parts[-1].split('.')[0]+'.txt'
    with open( str(topdir / outfileName), 'w+', encoding='utf8') as f:
        f.write(''.join(relex_sents) ) 


if __name__ == "__main__":
    plac.call(main)