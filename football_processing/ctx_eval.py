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
        if token.dep_ in ('ROOT'):
            rootV = token
        if token.pos_ in ('VERB'):
            verbs.append(token)

    return rootV, verbs


@plac.annotations(
    predfile = ('name of predfile', 'option', "p", Path),
    refile = ('path to ref file', 'option', 'r', Path),
    outfile = ('path to ouful', 'option', 'o', Path)
)
def main(predfile = '', refile ='', outfile= 'ctx_eval_out'):

    predictions = []
    with open( str(predfile) , 'r') as f:
        predictions = [line for line in f]

    references = []
    with open( str(refile) , 'r') as f:
        references = [line for line in f]

    assert len(predictions) == len(references), 'length of predictions and references do not match'

    nlp = load_nlp()

    scores = []

    for pred, refs in zip(predictions, references):

        score = 0
        pred_doc = nlp(pred)
        refs_doc = nlp(refs)

        pred_root, pred_verbs = filterSignificantVerbs(pred_doc)
        refs_root, refs_verbs = filterSignificantVerbs(refs_doc)

        if pred_root.lemma_ == refs_root.lemma_:
            score += 50

        if len(pred_verbs) == len(refs_verbs):
            score += 50

        scores.append(score)

    final_score = mean(scores)
    final_std = stdev(scores)

    with open( str(outfile) , 'w+') as f:
        for s in scores:
            f.write(str(s))
    print('CTX', final_score, '+-',final_std)



if __name__ == "__main__":
    plac.call(main)