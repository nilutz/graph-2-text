import plac
from pathlib import Path
import spacy


from statistics import mean, stdev

#https://github.com/DuyguA/DEMorphy/issues/4
#https://github.com/DuyguA/DEMorphy

#pip install ~/Dev/DEMorphy-master
#  cp ~/Dev/DEMorphy-master/demorphy/data/words.dg ~/Dev/graph-2-text/env/lib/python3.6/site-packages/demorphy/data 

from demorphy import Analyzer
from demorphy.cache import memoize, lrudecorator
analyzer = Analyzer(char_subs_allowed=True)
cache_size = 200 #you can arrange the size or unlimited cache. For German lang, we recommed 200 as cache size.
cached = memoize if cache_size=="unlim" else (lrudecorator(cache_size) if cache_size else (lambda x: x))
analyze = cached(analyzer.analyze)

DEBUG = False

def load_nlp():
    nlp = spacy.load('de_core_news_md', disable=['ner'])
    return nlp

def filterSignificantVerbs(doc):

    rootV = None
    verbs = []
    for token in doc:
        if token.dep_ == 'ROOT':
            rootV = token
        if token.pos_ in ('VERB', 'AUX'):
            verbs.append(token)
    return rootV, verbs

def check(token):

    if token is not None:
        return token.text
    return '-'


@plac.annotations(
    predfile = ('name of predfile', 'option', "p", Path),
    refile = ('path to ref file', 'option', 'r', Path),
    outfile = ('path to outfile', 'option', 'o', Path),
    fakefile = ('path to fake predictions', 'option', 'f', Path),
    ctxfile = ('path to ctx', 'option', 'c', Path),
    ctxfakefile = ('path to fake ctx', 'option', 'x', Path),
)
def main(predfile = '', refile ='', outfile= 'ctx_eval_out', fakefile = '', ctxfile='', ctxfakefile=''):
    '''
    a perfect score is 1.0
    '''
    # Load all the stuff
    
    predictions = []
    with open( str(predfile) , 'r') as f:
        predictions = [line for line in f]

    references = []
    with open( str(refile) , 'r') as f:
        references = [line for line in f]
        
    ctxs = []
    with open( str(ctxfile) , 'r') as f:
        for line in f:
            if 'report' in line:
                ctxs.append('past')
            else:
                ctxs.append('pres')
    
    fakes = []
    with open( str(fakefile) , 'r') as f:
        fakes = [line for line in f]
   
        
    ctxsfake = []
    with open( str(ctxfakefile) , 'r') as f:
        for line in f:
            if 'report' in line:
                ctxsfake.append('past')
            else:
                ctxsfake.append('pres') 
    
    assert len(predictions) == len(references), 'length of predictions and references do not match'

    nlp = load_nlp()

    scores = []
    scores_b = []
    words = []
    
    #make the score based on some rules
    for pred, refs, fakes, ctx, ctxfake in zip(predictions, references, fakes, ctxs, ctxsfake):

        score = 0 #with ref
        score_b = 0
        
        pred_doc = nlp(pred)
        refs_doc = nlp(refs)
        fakes_doc = nlp(fakes)
        

        pred_root, pred_verbs = filterSignificantVerbs(pred_doc)
        refs_root, refs_verbs = filterSignificantVerbs(refs_doc)
        fakes_root, fakes_verbs = filterSignificantVerbs(fakes_doc)

        if DEBUG:
            print('p=>', pred_doc, '=>',pred_root, pred_verbs, '?', ctx)
            print('f=>', fakes_doc, '=>',fakes_root, fakes_verbs,'?', ctxfake)
            print('ref=>', refs_doc, '=>',refs_root, refs_verbs, '?', ctx)
            
            
        refs = [r.lemma_ for r in refs_verbs]
        for verb in pred_verbs:
            a = analyze(verb.text)
            if verb.lemma_ in refs:
                if len(a) > 0 and a[0].tense == ctx:
                    score +=1

            if len(a) > 0 and a[0].tense == ctx:
                score_b +=1

        for verb in fakes_verbs:
            a = analyze(verb.text)
            if verb.lemma_ in refs:
                if len(a) > 0 and a[0].tense == ctxfake:
                    score +=1   

            if len(a) > 0 and a[0].tense == ctxfake:
                score_b +=1
        
        if len(refs_verbs) > 0 :
            score = score / (len(refs_verbs) * 2.)
        else:
            score = 0
        
        if len(pred_verbs) > 0 and len(fakes_verbs) > 0:
            score_b = score_b / (len(pred_verbs) + len(fakes_verbs))
        elif len(pred_verbs) > 0:
            score_b = score_b / len(pred_verbs)
        elif len(fakes_verbs) >0:
            score_b = score_b / len(fakes_verbs)
        else:
            score_b = 0
            
        
        
        if DEBUG: print('score(ref)=>',score, '-score: ',score_b,'\n')
        scores.append(score)
        scores_b.append(score_b)

        
        words.append([str(score), str(score_b), check(pred_root), check(refs_root), check(fakes_root) ])

    final_score = mean(scores)
    final_std = stdev(scores)
    
    final_score_b = mean(scores_b)
    final_std_b = stdev(scores_b)
    

    with open( str(outfile) , 'w+') as f:
        for s in scores:
            f.write(str(s)+'\n')

    fin2 = 'CTXE2 mean score (with ref):' + str(final_score) + ' +- ' + str(final_std)
    fin1 = 'CTXE1 mean score:' + str(final_score_b) + ' +- ' + str(final_std_b)


    with open( str(outfile).replace('.txt', '_details.txt') , 'w+') as f:
        for w in words:                
            f.write(' '.join(w)+'\n')
        f.write(fin1)
        f.write('\n')
        f.write(fin2)

    print('CTXE1 mean score:', final_score_b, '+-',final_std_b)
    print('CTXE2 mean score (with ref):', final_score, '+-',final_std)

if __name__ == "__main__":
    plac.call(main)