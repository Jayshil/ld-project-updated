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
from astropy.table import Table
from astropy.io import ascii
import cdspyreadme
import astropy.units as u


#path1 = input('Enter the path of this folder: ')
path1 = '/home/jayshil/Documents/Dissertation/ld-project-updated'

#table1 = open(path1 + '/Tables/ldc.dat','w')
#table2 = open(path1 + '/Tables/stellar.dat','w')
#table3 = open(path1 + '/Tables/planets.dat','w')

#---------------------------------------------------------------------------------------------
#--------------------------Taking Data from the data file-------------------------------------
#---------------------------------------------------------------------------------------------

name = np.loadtxt(path1 + '/data_new.dat', dtype = str, usecols = 0, unpack = True)
teff, mh, lg, p, pperr, pnerr, tc, aste, asteperr, astenerr, rprst, rprstperr, rprstnerr, tce1 = \
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
tce, tcep, tcen = np.loadtxt(path1 + '/Data/results.dat', usecols = (25,26,27), unpack = True)

#---------------------------------------------------------------------------------------------
#------------------------ Starting Iteration to calculate things------------------------------
#---------------------------------------------------------------------------------------------

"""
for i in range(len(teff)):
	table1.write(name[i][:-1] + ' & $' + str(np.format_float_positional(u1[i], 2)) + '^{+' + str(np.format_float_positional(u1p[i], 2)) + '}_{-' + str(np.format_float_positional(u1n[i], 2)) + '}$ & $' + str(np.format_float_positional(u2[i], 2)) + '^{+' + str(np.format_float_positional(u2p[i], 2)) + '}_{-' + str(np.format_float_positional(u2n[i], 2)) + '}$ & ' + str(np.format_float_positional(u1_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho_r[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho_r[i], 2)) + ' \\\\ \n\t')
	table2.write(name[i][:-1] + ' & ' + str(teff[i]) + ' & ' + str(lg[i]) + ' & ' + str(mh[i]) + ' & ... \\\\ \n\t')
	table3.write(name[i] + ' & $' + str(np.format_float_positional(re[i], 4)) + '^{+' + str(np.format_float_positional(reep[i], 4)) + '}_{-' + str(np.format_float_positional(reen[i], 4)) + '}$ & $' + str(np.format_float_positional(ae[i], 2)) + '^{+' + str(np.format_float_positional(aeep[i], 2)) + '}_{-' + str(np.format_float_positional(aeen[i], 2)) + '}$ & $' + str(np.format_float_positional(tce[i] - 2458000, 5)) + '^{+' + str(np.format_float_positional(tcep[i], 5)) + '}_{-' + str(np.format_float_positional(tcen[i], 5)) + '}$ & $' + str(np.format_float_positional(rprst[i], 4)) + '^{+' + str(np.format_float_positional(rprstperr[i], 4)) + '}_{-' + str(np.format_float_positional(rprstnerr[i], 4)) + '}$ & $' + str(np.format_float_positional(aste[i], 2)) + '^{+' + str(np.format_float_positional(asteperr[i], 2)) + '}_{-' + str(np.format_float_positional(astenerr[i], 2)) + '}$ & $' + str(np.format_float_positional(xx1[i] - 2458000, 3)) + ' \pm ' + str(np.format_float_positional(xx11[i], 3)) + '$ \\\\ \n\t')

table1.close()
table2.close()
table3.close()
"""

# For Stellar parameters
# for vturb
vt = []
for i in range(len(teff)):
	vt.append('...')
# for name
name1 = []
for i in range(len(teff)):
	name1.append(name[i][:-1])

# CDS table
#str(np.format_float_positional(aste[i], 2)) + '^{+' + str(np.format_float_positional(asteperr[i], 2)) + '}_{-' + str(np.format_float_positional(astenerr[i], 2)) + '}$ & $' + str(np.format_float_positional(xx1[i] - 2458000, 3)) + ' \pm ' + str(np.format_float_positional(xx11[i], 3)) + '$ \\\\ \n\t')

tab1 = Table()
tab1['name'], tab1['name'].info.format, tab1['name'].description = name, '%s', 'Name of the planet'
tab1['Rp/R*'], tab1['Rp/R*'].info.format, tab1['Rp/R*'].description = re, '%.4f', 'Rp/R* retrieved from this work'
tab1['Rp/R*_u'], tab1['Rp/R*_u'].info.format, tab1['Rp/R*_u'].description = reep, '%.4f', 'Upper 68% credibility band on Rp/R*'
tab1['Rp/R*_l'], tab1['Rp/R*_l'].info.format, tab1['Rp/R*_l'].description = reen, '%.4f', 'Lower 68% credibility band on Rp/R*'
tab1['a/R*'], tab1['a/R*'].info.format, tab1['a/R*'].description = ae, '%.2f', 'a/R* retrieved from this work'
tab1['a/R*_u'], tab1['a/R*_u'].info.format, tab1['a/R*_u'].description = aeep, '%.2f', 'Upper 68% credibility band on a/R*'
tab1['a/R*_l'], tab1['a/R*_l'].info.format, tab1['a/R*_l'].description = aeen, '%.2f', 'Lower 68% credibility band on a/R*'
tab1['tc'], tab1['tc'].info.format, tab1['tc'].description = tce, '%.5f', 'Transit time (BJD - 2458000) from this work'
tab1['tcep'], tab1['tcep'].info.format, tab1['tcep'].description = tcep, '%.5f', 'Upper 68% credibility band on transit time'
tab1['tcen'], tab1['tcen'].info.format, tab1['tcen'].description = tcen, '%.5f', 'Lower 68% credibility band on transit time'
tab1['rprst'], tab1['rprst'].info.format, tab1['rprst'].description = rprst, '%.4f', 'Literature value of Rp/R*'
tab1['rprst_ue'], tab1['rprst_ue'].info.format, tab1['rprst_ue'].description = rprstperr, '%.4f', 'Upper 68% credibility on Rp/R* from literature'
tab1['rprst_le'], tab1['rprst_le'].info.format, tab1['rprst_le'].description = rprstnerr, '%.4f', 'Lower 68% credibility on Rp/R* from literature'
tab1['aste'], tab1['aste'].info.format, tab1['aste'].description = aste, '%.2f', 'Literature value of a/R*'
tab1['aste_ue'], tab1['aste_ue'].info.format, tab1['aste_ue'].description = asteperr, '%.2f', 'Upper 68% credibility band on literature value of a/R*'
tab1['aste_le'], tab1['aste_le'].info.format, tab1['aste_le'].description = astenerr, '%.2f', 'Lower 68% credibility band on literature value of a/R*'


tab1.write(os.getcwd() + '/Tables/planetary_ascii.dat', format='ascii.mrt', overwrite=True, delimiter='\t\t')
#	 formats={'name':'%20s', 'temp':'%4.1f', 'logg':'%1.3f', 'mh':'%1.2f', 'vturb':'%s'})