import plac
from pathlib import Path
import spacy


from statistics import mean, stdev

def load_nlp():
    nlp = spacy.load('de_core_news_md', disable=['ner'])
    return nlp

def filterSignificantVerbs(doc):

    rootV = None
    verbs = []
    for token in doc:
        if token.dep_ == 'ROOT':
            print('ROOT',token)
            rootV = token
        if token.pos_ == 'VERB':
            verbs.append(token)

    return rootV, verbs


@plac.annotations(
    predfile = ('name of predfile', 'option', "p", Path),
    refile = ('path to ref file', 'option', 'r', Path),
    outfile = ('path to ouful', 'option', 'o', Path)
    fakefile = ('path to fake predictions', 'options', 'f', Path)
)
def main(predfile = '', refile ='', outfile= 'ctx_eval_out', fakefile = ''):

    predictions = []
    with open( str(predfile) , 'r') as f:
        predictions = [line for line in f]

    references = []
    with open( str(refile) , 'r') as f:
        references = [line for line in f]
    
    fakes = []
    if fakefile != '':
        with open( str(fakefile) , 'r') as f:
            fakes = [line for line in f]
    else:
        fakes = ['_' for _ in predictions] #init empty

    assert len(predictions) == len(references), 'length of predictions and references do not match'

    #nlp = load_nlp()

    scores = []

    words = []

    for pred, refs in zip(predictions, references, fakes):

        score = 0
        pred_doc = nlp(pred)
        refs_doc = nlp(refs)
        fakes_doc = nlp(fakes)
        


        pred_root, pred_verbs = filterSignificantVerbs(pred_doc)
        refs_root, refs_verbs = filterSignificantVerbs(refs_doc)
        fakes_root, fakes_verbs = filterSignificantVerbs(fakes_doc)


        print('p=>', pred_doc, '=>',pred_root, pred_verbs)
        print('r=>', refs_doc, '=>',refs_root, refs_verbs)
        print('f=>', fakes_doc, '=>',fakes_root, fakes_verbs)


        if pred_root.lemma_ == refs_root.lemma_: #same lemma and ROOT 
            score += 5
        
        if pred_root.text == refs_root.text: #same ROOT and TENSE -> best case
            score += 45

        if len(pred_verbs) == len(refs_verbs): #same number of verbs
            score += 5
            nomatch = set(pred_verbs) & set(refs_verbs) #all are the same -> best case
            if len(nomatch) == 0:
                score += 45
        else:
            rv = [r.text for r in refs_verbs]
            for p in pred_verbs:
                print(p)
                if p.text in rv: #get score of if at least some preds are corrects
                    score +=5 
                
        scores.append(score)
        words.append([str(score), pred_root.text, refs_root.text ])

    final_score = mean(scores)
    final_std = stdev(scores)

    with open( str(outfile) , 'w+') as f:
        for s in scores:
            f.write(str(s)+'\n')

    with open( str(outfile).replace('.txt', '_details.txt') , 'w+') as f:
        for w in words:                
            f.write(' '.join(w)+'\n')

    print('CTX mean score:', final_score, '+-',final_std)

if __name__ == "__main__":
    plac.call(main)