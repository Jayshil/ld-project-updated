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
bb, bbp, bbn = np.loadtxt(path1 + '/Data/results.dat', usecols = (7,8,9), unpack=True)
per, perp, pern = np.loadtxt(path1 + '/Data/results.dat', usecols = (28,29,30), unpack=True)

#---------------------------------------------------------------------------------------------
#------------------------ Starting Iteration to calculate things------------------------------
#---------------------------------------------------------------------------------------------
# for ticids
nm1 = np.loadtxt(os.getcwd() + '/Tables/ticids.dat', usecols=0, unpack=True, dtype=str)
tic1 = np.loadtxt(os.getcwd() + '/Tables/ticids.dat', usecols=1, unpack=True)

ticid = np.zeros(len(name))
for i in range(len(name)):
	for j in range(len(nm1)):
		if name[i] == nm1[j]:
			ticid[i] = tic1[j]

# CDS table
#str(np.format_float_positional(xx1[i] - 2458000, 3)) + ' \pm ' + str(np.format_float_positional(xx11[i], 3)) + '$ \\\\ \n\t')

tab1 = Table()
tab1['name'], tab1['name'].info.format, tab1['name'].description = name, '%s', 'Name of the planet'
tab1['tic'], tab1['tic'].info.format, tab1['tic'].description = ticid, '%d', 'TIC id of the planet'
tab1['Rp/R*'], tab1['Rp/R*'].info.format, tab1['Rp/R*'].description = re, '%.4f', 'Rp/R* retrieved from this work'
tab1['Rp/R*_u'], tab1['Rp/R*_u'].info.format, tab1['Rp/R*_u'].description = reep, '%.4f', 'Upper 68% credibility band on Rp/R*'
tab1['Rp/R*_l'], tab1['Rp/R*_l'].info.format, tab1['Rp/R*_l'].description = reen, '%.4f', 'Lower 68% credibility band on Rp/R*'
tab1['a/R*'], tab1['a/R*'].info.format, tab1['a/R*'].description = ae, '%.2f', 'a/R* retrieved from this work'
tab1['a/R*_u'], tab1['a/R*_u'].info.format, tab1['a/R*_u'].description = aeep, '%.2f', 'Upper 68% credibility band on a/R*'
tab1['a/R*_l'], tab1['a/R*_l'].info.format, tab1['a/R*_l'].description = aeen, '%.2f', 'Lower 68% credibility band on a/R*'
tab1['tc'], tab1['tc'].info.format, tab1['tc'].description, tab1['tc'].unit = tce, '%.5f', 'Transit time (in BJD) from this work', u.day
tab1['tcep'], tab1['tcep'].info.format, tab1['tcep'].description, tab1['tcep'].unit = tcep, '%.5f', 'Upper 68% credibility band on transit time', u.day
tab1['tcen'], tab1['tcen'].info.format, tab1['tcen'].description, tab1['tcen'].unit= tcen, '%.5f', 'Lower 68% credibility band on transit time', u.day
tab1['b'], tab1['b'].info.format, tab1['b'].description = bb, '%.3f', 'b retrieved from this work'
tab1['b_u'], tab1['b_u'].info.format, tab1['b_u'].description = bbp, '%.3f', 'Upper 68% credibility band on b'
tab1['b_l'], tab1['b_l'].info.format, tab1['b_l'].description = bbn, '%.3f', 'Lower 68% credibility band on b'
tab1['P'], tab1['P'].info.format, tab1['P'].description, tab1['P'].unit = per, '%.7f', 'period retrieved from this work', u.day
tab1['P_u'], tab1['P_u'].info.format, tab1['P_u'].description, tab1['P_u'].unit = perp, '%.7f', 'Upper 68% credibility band on period', u.day
tab1['P_l'], tab1['P_l'].info.format, tab1['P_l'].description, tab1['P_l'].unit = pern, '%.7f', 'Lower 68% credibility band on period', u.day

tab1.write(os.getcwd() + '/Tables/planetary_ascii_updated.dat', format='ascii.mrt', overwrite=True, delimiter='\t\t')
#	 formats={'name':'%20s', 'temp':'%4.1f', 'logg':'%1.3f', 'mh':'%1.2f', 'vturb':'%s'})
