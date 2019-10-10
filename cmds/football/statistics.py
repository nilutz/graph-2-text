import plac
import glob
from pathlib import Path
import numpy as np
import re

@plac.annotations(
    topdir = ("dir to predfile", "option", "t", str),
    experiment = ("experiment  number ","option", "e", int),
)
def main(topdir = '.', experiment = 1):

    if experiment == 1:
        types_ = ['notdelex_attr_', 'notdelex_attr_postfix_','delex_attr_', 'delex_attr_postfix_', 'delex_attr_postfix_simple_']
        
        nums = [str(i) for i in range(1,4)]

    elif experiment == 2:
        types_ = ['hyper_dense_256_3_1_c', 'hyper_dense_256_4_1_c', 'hyper_dense_256_5_1_c',
                 'hyper_dense_512_3_1_c', 'hyper_dense_512_4_1_c', 'hyper_dense_512_5_1_c',
                 'hyper_dense_1024_3_1_', 'hyper_dense_1024_4_1_', 'hyper_dense_1024_5_1_',
                 'hyper_residual_256_3_', 'hyper_residual_256_4_', 'hyper_residual_256_5_',
                 'hyper_residual_512_3_', 'hyper_residual_512_4_', 'hyper_residual_512_5_',
                 'hyper_residual_1024_3', 'hyper_residual_1024_4', 'hyper_residual_1024_5',
        ]
        nums = [str(i) for i in range(1)]

    elif experiment == 3:

        types_ = ['delex_attr_postfix_']
        nums = [str(i) for i in range(4,7)]
    else:
        raise "UNDEFINED EXPERIMENT"

    print('AVG Metrics for experiment: ', experiment)
    for types in types_:
        bleus=[]
        ters=[]
        meteors=[]
        ctx1 = []
        ctx2 = []

        for n in nums:

            if experiment == 1 or 3:
                filenameb = topdir + "/out_bleu_"+types+n+'.txt'
                filenamet = topdir + "/out_ter_"+types+n+'.txt'
                filenamem = topdir + "/out_meteor_"+types+n+'.txt'
                filenamec = '../../data/data-football/'+types+'/ctx_eval_'+n+'_details.txt'
            else:
                filenameb = topdir + "/out_bleu_delex_attr_postfix_"+types+'.txt'
                filenamet = topdir + "/out_ter_delex_attr_postfix_"+types+'.txt'
                filenamem = topdir + "/out_meteor_delex_attr_postfix_"+types+'.txt'
                filenamec = '../../data/data-football/delex_attr_postfix_/ctx_eval_delex_attr_postfix_'+types+'_details.txt'
  
            with open( str(filenameb) , 'r') as f:
                for line in f:
                    if 'BLEU' in line:
                        bleus.append(float(re.sub(',','',line.split()[2]) ) )
            with open( str(filenamet) , 'r') as f:
                for line in f:
                    if 'Total TER:' in line:
                        ters.append(float(re.sub(',','',line.split()[2]) ) )
            with open( str(filenamem) , 'r') as f:
                for line in f:
                    if 'Final score:' in line:
                        meteors.append(float(re.sub(',','',line.split()[2]) ) )

            with open( str(filenamec) , 'r') as f:
                for line in f:
                    if 'CTXE1 mean score:' in line:
                        ctx1.append(float(re.sub('score:','',line.split()[2]) ) )
                    if 'CTXE2' in line:
                        ctx2.append(float(re.sub(r'ref\):','',line.split()[4] ) ) )

        print(types)
        print('BLEU   MEAN: ', np.mean(bleus), '  std: ', np.std(bleus), bleus)
        print('METEOR MEAN: ', np.mean(meteors), '  std: ',np.std(meteors), meteors)
        print('TER    MEAN: ', np.mean(ters), ' std: ',np.std(ters), ters)
        print('CTXE1  MEAN: ', np.mean(ctx1), ' std: ',np.std(ctx1), ctx1)
        print('CTXE2  MEAN: ', np.mean(ctx2), ' std: ',np.std(ctx2), ctx2)        
        print('\n')
        
if __name__ == "__main__":
    plac.call(main)

