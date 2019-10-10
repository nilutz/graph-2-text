import plac
import glob
from pathlib import Path
import numpy as np
import re

@plac.annotations(
    topdir = ("dir to predfile", "option", "t", str),
)
def main(topdir = '.'):

    print('AVG Metrics for webnlg: ')
    bleus=[]
    ters=[]
    meteors=[]
    ctx1 = []
    ctx2 = []

    for n in nums:

        filenameb = topdir + "/out_bleu_"+n+'.txt'
        filenamet = topdir + "/out_ter_"+n+'.txt'
        filenamem = topdir + "/out_meteor_"+n+'.txt'
           
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

        print('BLEU   MEAN: ', np.mean(bleus), '  std: ', np.std(bleus), bleus)
        print('METEOR MEAN: ', np.mean(meteors), '  std: ',np.std(meteors), meteors)
        print('TER    MEAN: ', np.mean(ters), ' std: ',np.std(ters), ters)
     
        print('\n')
        
if __name__ == "__main__":
    plac.call(main)

