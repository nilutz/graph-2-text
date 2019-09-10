import pandas as pd
import networkx as nx
from pathlib import Path
import plac
import csv
import numpy as np
import os
import random
import re
import json

# import matplotlib.pyplot as plt

# def draw(DG):

#     pos = nx.spring_layout(DG, scale=0.5)

#     node_labels = dict((n,n) for n,d in DG.nodes(data=True))
#     edge_labels = dict(((u,v),list(d.values())[0]) for u,v,d in DG.edges(data=True))

#     nx.draw(DG, pos=pos, alpha=0.8, arrows=False, node_color='lightgrey', node_size=400,
#             labels=node_labels, 
#             font_color='black', font_size=8, font_weight='bold',
#            )
#     nx.draw_networkx_edge_labels(DG, pos, edge_labels = edge_labels, font_size=8)
#     plt.show()


def genMultiGraph(DG, verbose=False):
    #In here we adopt the parametrization proposed by Marcheggiani and Titov (2017) where edge labels and directions are explicitly modeled.
    #https://arxiv.org/pdf/1703.04826.pdf
    #SUBJ(AO) -> rel -> (A1)obj

    srcNodes = []
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []
    
    for eTriple in DG.edges(data='label'):
        rel = "_".join([x.strip() for x in eTriple[2].split()])
        subj = [x.strip() for x in eTriple[0].split()]
        obj = [x.strip() for x in eTriple[1].split()]
        if verbose: print(subj, rel, obj)

        subjNodeDescendants = []
        objNodeDescendants = []
        subjNode = subj[0]
        if len(subj) > 1:
            subjNodeDescendants = subj[1:]
        objNode = obj[0]
        if len(obj):
            objNodeDescendants = obj[1:

                                 ]
        if not subjNode in srcNodes:
            srcNodes.append(subjNode)
        srcNodes.append(rel)
        relIdx = len(srcNodes) - 1
        if not objNode in srcNodes:
            srcNodes.append(objNode)

        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(srcNodes.index(subjNode)))
        srcEdgesNode2.append(str(relIdx))

        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(srcNodes.index(objNode)))
        srcEdgesNode2.append(str(relIdx))

        if subjNodeDescendants:
            for neNode in subjNodeDescendants:
                srcNodes.append(neNode)
                nodeIdx = len(srcNodes) -1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(srcNodes.index(subjNode)))

        if objNodeDescendants:
            for neNode in objNodeDescendants:
                srcNodes.append(neNode)
                nodeIdx = len(srcNodes) -1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(srcNodes.index(objNode)))


    if verbose:
        print(" ".join(srcNodes))
        print(len(srcNodes))
        print('\n')

        print(" ".join(srcEdgesLabels))
        print(len(srcEdgesLabels))
        print('\n')

        print(" ".join(srcEdgesNode1))
        print(len(srcEdgesNode1))
        print('\n')

        print(" ".join(srcEdgesNode2))
        print(len(srcEdgesNode2))
        print('\n')
    return " ".join(srcNodes), (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2))


def check_upper(s):
    if not s.isupper():
        return s.lower()
    else:
        return s
    
def preprocess_triples(df, options, classtype = '', ctx = True):
    '''
    reads in the texts and triples in different options
    and transforms into the parametrization proposed by Marcheggiani and Titov
    '''
    
    source_out = []
    source_nodes_out = []
    source_edges_out_labels = []
    source_edges_out_node1 = []
    source_edges_out_node2 = []
    target_out = []
    context = []
    relex = []
    reference = []

    count = 0
    for i in range(0, len(df)):

       
        try:
            triples = df.at[i,'triples']
            if type(triples) is float:
                continue
        except:
            #print('some error')
            continue
        #no empty triples !!!
        if triples is None or len(triples)==0:
            continue
        count += 1
        
        text = df.at[i,'text']
        if options['lower'] and options['notdelex']:
            text = df.at[i,'text'].lower()
        elif options['lower'] and options['notdelex'] == False:
            lower = []
            try:
                for word in df.at[i,'text'].split():
                    if word.isupper():
                        lower.append(word)
                    else:
                        lower.append(word.lower())
                text = ' '.join(lower)
            except:
                continue
        #else:
        #    print('No options specified')


        if type(text) is float:
            continue

        DG = nx.MultiDiGraph()
        
        tripleSep = ""
        triplesString = ''
        for triple in triples:
            
            if options['lower']:

                DG.add_edge(check_upper(triple[0]), check_upper(triple[2]), label = check_upper(triple[1]))
                triplesString += tripleSep + check_upper(triple[0]) + '|' + check_upper(triple[1]) + '|' + check_upper(triple[2]) + ' '

            else:
                DG.add_edge(triple[0], triple[2], label = triple[1])
                triplesString += tripleSep + triple[0] + '|' + triple[1] + '|' + triple[2] + ' '
            tripleSep = "<TSP>"
        
        source_nodes, source_edges = genMultiGraph(DG)
        
        source_nodes_out.append(source_nodes)
        source_edges_out_labels.append(source_edges[0])
        source_edges_out_node1.append(source_edges[1])
        source_edges_out_node2.append(source_edges[2])
        target_out.append(text)
        out_src = ' '.join(re.split('(\W)', triplesString))
        source_out.append(' '.join(out_src.split()))
        

        if ctx:
            context.append(df.at[i,'class'])

        if options['notdelex'] == False:
            relex.append(json.dumps(df.at[i, 'relexDict']))
            reference.append(df.at[i,'referenceText'])

    if options['notdelex'] == False:
        concat = list(zip(source_nodes_out, source_edges_out_labels, source_edges_out_node1, source_edges_out_node2, target_out, context, source_out, relex, reference))
        random.Random(42).shuffle(concat)
        source_nodes_out, source_edges_out_labels, source_edges_out_node1, source_edges_out_node2, target_out, context, source_out, relex, reference  = zip(*concat)

    else:  
        concat = list(zip(source_nodes_out, source_edges_out_labels, source_edges_out_node1, source_edges_out_node2, target_out, context, source_out))
        random.Random(42).shuffle(concat)
        source_nodes_out, source_edges_out_labels, source_edges_out_node1, source_edges_out_node2, target_out, context, source_out  = zip(*concat)

    split_1 = int(0.8 * len(source_nodes_out))
    split_2 = int(0.9 * len(source_nodes_out))
        
    dataset='data-football'
    optionals = ''
    for key,val in options.items():
        if key == 'notdelex' and val == True:
            optionals+= 'notdelex'
        elif key =='notdelex' and val ==False:
            optionals += "delex"

        if val == True and key != "notdelex":
            optionals += '_'+key


    p = '../data/' + dataset +'/'+optionals+'_'+classtype

    for split in ('train', 'dev', 'test', 'test_fake'):

        if split == 'train':
            src = source_nodes_out[:split_1]
            edges_labels = source_edges_out_labels[:split_1]
            edges_node1 = source_edges_out_node1[:split_1]
            edges_node2 = source_edges_out_node2[:split_1]
            tgt_out = target_out[:split_1]
            src_out = source_out[:split_1]
            if ctx:
                context_out = context[:split_1]
            if options['notdelex'] == False:
                relex_out = relex[:split_1]
                ref_out = reference[:split_1]


        elif split == 'dev':
            src = source_nodes_out[split_1:split_2]
            edges_labels = source_edges_out_labels[split_1:split_2]
            edges_node1 = source_edges_out_node1[split_1:split_2]
            edges_node2 = source_edges_out_node2[split_1:split_2]
            tgt_out = target_out[split_1:split_2]
            src_out = source_out[split_1:split_2]

            if ctx:
                context_out = context[split_1:split_2]
            if options['notdelex'] == False:
                relex_out = relex[split_1:split_2]
                ref_out = reference[split_1:split_2]

        elif split == 'test':
            src = source_nodes_out[split_2:]
            edges_labels = source_edges_out_labels[split_2:]
            edges_node1 = source_edges_out_node1[split_2:]
            edges_node2 = source_edges_out_node2[split_2:]
            tgt_out = target_out[split_2:]
            src_out = source_out[split_2:]
            if ctx:
                context_out = context[split_2:]
            if options['notdelex'] == False:
                relex_out = relex[split_2:]
                ref_out = reference[split_2:]

                
        #create fake test set by invering the context
        elif split == 'test_fake' and ctx:
            src = source_nodes_out[split_2:]
            edges_labels = source_edges_out_labels[split_2:]
            edges_node1 = source_edges_out_node1[split_2:]
            edges_node2 = source_edges_out_node2[split_2:]
            tgt_out = target_out[split_2:]
            if ctx:
                context_out = context[split_2:]
                context_fake =[]
                for c in context_out:
                    if c == 'ticker':
                        context_fake.append('report')
                    else:
                        context_fake.append('ticker')

                context_out = context_fake

        if not os.path.exists(p):
            os.makedirs(p)

        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-src-nodes.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(src))
            
        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-src-labels.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(edges_labels))
        
        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-src-node1.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(edges_node1))
        
        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-src-node2.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(edges_node2))
        
        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-tgt.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(tgt_out)) 

        with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '.triple', 'w+', encoding='utf8') as f:
            f.write('\n'.join(src_out))
        
        if ctx:
            with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '-context.txt', 'w+', encoding='utf8') as f:
                f.write('\n'.join(context_out)) 
        
        if options['notdelex'] == False:
            with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '.relex', 'w+', encoding='utf8') as f:
                f.write('\n'.join(relex_out)) 
            with open(p+'/' + split + '-' +dataset + '-gcn-' + optionals + '.reference', 'w+', encoding='utf8') as f:
                f.write('\n'.join(ref_out))
        
        assert len(tgt_out) == len(edges_node1)
        assert len(src) == len(edges_node1)
        assert len(edges_node2) == len(edges_node1)
        assert len(edges_node2) == len(edges_labels)
        if ctx:
            assert len(src) == len(context_out)

    print('processed in process triple', len(df), count, len(target_out) ,len(source_nodes_out),len(source_edges_out_node1), len(source_edges_out_node2),len(source_edges_out_labels),'texts with',optionals)

    optionals = optionals
    return p, optionals, dataset

def split_to_csv(p, dataset = 'football', options='', ctx = False):
    '''
    makes a train / test / val split in a  0.8 / 0.1 / 0.1 ratio
    
    '''
    for split in ('train', 'dev', 'test'):

        tgt_corpus = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-tgt.txt',header=None, delimiter='\n', names=['tgt'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        label_corpus = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-src-labels.txt',header=None, delimiter='\n', names=['label'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        node1_corpus = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-src-node1.txt', header=None,delimiter='\n', names=['node1'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        node2_corpus = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-src-node2.txt',header=None, delimiter='\n', names=['node2'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        nodes_corpus = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-src-nodes.txt',header=None, delimiter='\n', names=['nodes'], quoting=csv.QUOTE_NONE, error_bad_lines=False)

        if ctx:
            context = pd.read_csv(p+'/'+ split + '-' + dataset + '-gcn-' + options +'-context.txt',header=None, delimiter='\n', names=['ctx'], quoting=csv.QUOTE_NONE, error_bad_lines=False)

        print('processed in split', len(tgt_corpus) ,len(label_corpus),len(node1_corpus), len(node2_corpus),len(nodes_corpus),'texts')
        assert len(tgt_corpus) == len(nodes_corpus)

        df = pd.concat([tgt_corpus,label_corpus,node1_corpus, node2_corpus, nodes_corpus],axis = 1, sort=False)
        if ctx:
            df = pd.concat([df, context], axis = 1, sort = False)
        # print(len(df))
        # train, validate, test = np.split(df.sample(frac=1, random_state=1), [int(.8*len(df)), int(.9*len(df))])
        
        if not os.path.exists(p):
            os.makedirs(p)
        df.to_csv(p+'/'+split+'.csv', encoding='UTF-8')
            
        #train.to_csv(p+'/'+'train.csv', encoding='UTF-8')
        #test.to_csv(p+'/'+'test.csv', encoding='UTF-8')
        #validate.to_csv(p+'/'+'val.csv', encoding='UTF-8')

@plac.annotations(
    path = ("path to df for preprocessing", "option", "p", Path),
    ctx = ('Also process context',"flag", 'c' ),
    full = ('Full df or split by class', "flag", "f"),
)
def main(path = "../data/data-football/sentences_full_notdelex.pkl", ctx=False, full = False):

    #we inspect the path to find out what options we have here
    parts = path.parts[-1].split('_')
    options = {
        'notdelex': False, #notdelex
        'lower': False,
    }
    for p in parts:
        if 'notdelex' in p:
            options['notdelex'] = True
        if 'lower' in p:
            options['lower'] = True

    parts = path.parts[-1].split('_')
    print(parts)
    for p in parts:
        if 'comma' in p:
            options['comma'] = True
        if 'fre' in p:
            options['fre'] = True
    print(options)

    
    #df = pd.read.csv(path)
    df = pd.read_pickle(path)

    print('Using context', ctx)
    #makes tgt, nodesc,labels
    if not full:
        print('Splitting into ticker and report')
        mask = df['class'] == 'ticker'
        df1 = df[mask]
        df2 = df[~mask]

        df1.reset_index(inplace=True, drop=True)
        df2.reset_index(inplace=True, drop=True)

        p, options, dataset = preprocess_triples(df1, options=options, classtype='ticker', ctx=ctx)
        split_to_csv(p, dataset=dataset, options=options, ctx=ctx)
        
        p, options, dataset = preprocess_triples(df2, options=options, classtype='report', ctx=ctx)
        split_to_csv(p, dataset=dataset, options=options, ctx=ctx)

    else:
        p, options, dataset  = preprocess_triples(df, options=options, ctx=ctx)
        split_to_csv(p, dataset=dataset, options=options, ctx=ctx)

    #makes. train/val/test split an saves to .txt file

    #football = types='/football-gcn-'
    #webnlg  = types = 'train-webnlg-all-delex'


    # triples = df.at[1,'triples']
    # text = df.at[1,'text']

    # viz = True


    # DG = nx.MultiDiGraph()


    # for triple in triples:
    #     DG.add_edge(triple[0],triple[2], label = triple[1])

    # if viz:
    #     draw(DG)
    # print(text)
    # for eTriple in DG.edges(data='label'):
    #     rel = "_".join([x.strip() for x in eTriple[2].split()]) #eTriple[2].replace(" ", "_")
    #     subj = "_".join([x.strip() for x in eTriple[0].split()]) #eTriple[0].replace(" ", "_")
    #     obj = "_".join([x.strip() for x in eTriple[1].split()]) #eTriple[1].replace(" ", "_")
        
    #     print(subj, rel, obj)

if __name__ == "__main__":
    plac.call(main)
