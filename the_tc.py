import numpy as np
import os

# To compute theoretical Transit central time

f1 = open(os.getcwd() + '/Results/comp_a_r_p/to_the.dat','w')

# Literature data
name = np.loadtxt('data_new.dat', dtype = str, usecols = 0, unpack = True)
teff, lg, mh, p, pperr, tc, aste, asteperr, rprst, rprstperr, tce =\
	 np.loadtxt('data_new.dat', usecols = (9,11,10,1,2,3,5,6,7,8,4), unpack = True)
vturb = np.ones(len(teff))*2.0

# Results
t0, t0e = np.loadtxt(os.getcwd() + '/Data/results.dat', usecols=(25, 26), unpack=True)

for i in range(len(t0)):
    nn = np.around((t0[i]-tc[i])/p[i], 0)
    nn2 = int(nn)
    tinit = np.random.normal(tc[i], tce[i], 10000)
    perd = np.random.normal(p[i], pperr[i], 10000)
    tfin = tinit + (nn2*perd)
    f1.write(str(name[i]) + '\t' + str(np.mean(tfin)) + '\t' + str(np.std(tfin)) + '\n')

f1.close()