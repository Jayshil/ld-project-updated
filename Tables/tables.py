import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gd
import os
import pickle as pc
#import utils1 as utl
from astropy.io import fits
from astroquery.mast import Observations as obs
from scipy import interpolate as inp
import matplotlib.cm as cm
import matplotlib.colors as cls
from scipy.optimize import curve_fit as cft


#path1 = input('Enter the path of this folder: ')
path1 = '/home/jayshil/Documents/Dissertation/ld-project-updated'

table1 = open(path1 + '/Tables/ldc.dat','w')
table2 = open(path1 + '/Tables/stellar.dat','w')
table3 = open(path1 + '/Tables/planets.dat','w')

#---------------------------------------------------------------------------------------------
#--------------------------Taking Data from the data file-------------------------------------
#---------------------------------------------------------------------------------------------

name = np.loadtxt(path1 + '/data_new.dat', dtype = str, usecols = 0, unpack = True)
teff, lg, mh, p, pperr, pnerr, tc, aste, asteperr, astenerr, rprst, rprstperr, rprstnerr, tce1 = \
    np.loadtxt(path1 + '/data_new.dat', usecols = (9, 10, 11, 1, 2, 2, 3, 5, 6, 6, 7, 8, 8, 4), unpack = True)

#cite = np.loadtxt('citations', usecols=1, dtype=str, unpack=True)

xx1, xx11 = np.loadtxt(path1 + '/Results/comp_a_r_p/to_the.dat', usecols = (1,2), unpack = True)

u1_code_ata, u2_code_ata = np.loadtxt(path1 + '/Atlas/code_limiting_LDC_ata.dat', usecols = (1,2), unpack = True)
u1_cla_ata, u2_cla_ata = np.loadtxt(path1 + '/Atlas/claret_limiting_LDC_ata.dat', usecols = (1,2), unpack = True)

u1_code_pho, u2_code_pho = np.loadtxt(path1 + '/Phoenix/code_limiting_LDC_pho.dat', usecols = (1,2), unpack = True)
u1_cla_pho, u2_cla_pho = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho.dat', usecols = (1,2), unpack = True)
u1_cla_pho_r, u2_cla_pho_r = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho_r.dat', usecols = (1,2), unpack = True)

u1, u1p, u1n, u2, u2p, u2n = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True)

re, reep, reen = np.loadtxt(path1 + '/Data/results.dat', usecols = (4,5,6), unpack = True)
ae, aeep, aeen = np.loadtxt(path1 + '/Data/results.dat', usecols = (22,23,24), unpack = True)
tce, tcep, tcen = np.loadtxt(path1 + '/Results/comp_a_r_p/to_the.dat', usecols = (1,2,2), unpack = True)

#---------------------------------------------------------------------------------------------
#------------------------ Starting Iteration to calculate things------------------------------
#---------------------------------------------------------------------------------------------

for i in range(len(teff)):
	table1.write(name[i][:-1] + ' & $' + str(np.format_float_positional(u1[i], 2)) + '^{+' + str(np.format_float_positional(u1p[i], 2)) + '}_{-' + str(np.format_float_positional(u1n[i], 2)) + '}$ & $' + str(np.format_float_positional(u2[i], 2)) + '^{+' + str(np.format_float_positional(u2p[i], 2)) + '}_{-' + str(np.format_float_positional(u2n[i], 2)) + '}$ & ' + str(np.format_float_positional(u1_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho_r[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho_r[i], 2)) + ' \\\\ \n\t')
	table2.write(name[i][:-1] + ' & ' + str(teff[i]) + ' & ' + str(lg[i]) + ' & ' + str(mh[i]) + ' & ... \\\\ \n\t')
	table3.write(name[i] + ' & $' + str(np.format_float_positional(re[i], 3)) + '^{+' + str(np.format_float_positional(reep[i], 3)) + '}_{-' + str(np.format_float_positional(reen[i], 3)) + '}$ & $' + str(np.format_float_positional(ae[i], 3)) + '^{+' + str(np.format_float_positional(aeep[i], 3)) + '}_{-' + str(np.format_float_positional(aeen[i], 3)) + '}$ & $' + str(np.format_float_positional(tce[i] - 2458000, 5)) + '^{+' + str(np.format_float_positional(tcep[i], 5)) + '}_{-' + str(np.format_float_positional(tcen[i], 5)) + '}$ & $' + str(np.format_float_positional(rprst[i], 3)) + '^{+' + str(np.format_float_positional(rprstperr[i], 3)) + '}_{-' + str(np.format_float_positional(rprstnerr[i], 3)) + '}$ & $' + str(np.format_float_positional(aste[i], 2)) + '^{+' + str(np.format_float_positional(asteperr[i], 2)) + '}_{-' + str(np.format_float_positional(astenerr[i], 2)) + '}$ & $' + str(np.format_float_positional(xx1[i] - 2458000, 3)) + ' \pm ' + str(np.format_float_positional(xx11[i], 3)) + '$ \\\\ \n\t')

table1.close()
table2.close()
table3.close()