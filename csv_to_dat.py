import numpy as np
import os

## A Very short script to convert literature data
## ,which is in .csv format,
## into ascii data file

#Name	#Per	#Per_err	#T0	#T0_err	#a/R*	#a/R*_err	#Rp/R*	#Rp/R*_err	#Teff	#M/H	#logg	#TIC    #T0_ref #aR_ref #Rp/R*_ref  #Teff_ref   #Met_ref    #logg_ref
# 0     1       2           3   4       5       6           7       8           9       10      11      12      13      14      15          16          17          18

data = np.genfromtxt(os.getcwd() + '/Data/lit_data.csv', delimiter=',', dtype=str)

f1 = open(os.getcwd() + '/data_new.dat', 'w')
f1.write('#Name\tPer\tPer_err\tTc\tTc_err\ta/R*\ta/R*_err\tRp/R*\tRp/R*_err\tTeff\tMH\tlogg\tT0_ref\taR_ref\tRpR*_ref\tTeff_ref\tMet_ref\tlogg_ref\n')

for i in range(len(data[:,0])):
    f1.write(data[i][0] + '\t' + data[i][1] + '\t' + data[i][2] + '\t' + data[i][3] + '\t' + data[i][4]\
         + '\t' + data[i][5] + '\t' + data[i][6] + '\t' + data[i][7] + '\t' + data[i][8] + '\t' + data[i][9]\
         + '\t' + data[i][10] + '\t' + data[i][11] + '\t' + data[i][13].split('/')[4] + '\t' + data[i][14].split('/')[4]\
         + '\t' + data[i][15].split('/')[4] + '\t' + data[i][16].split('/')[4] + '\t' + data[i][17].split('/')[4] + '\t' + data[i][18].split('/')[4] + '\n')

f1.close()