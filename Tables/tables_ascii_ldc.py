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

u1_code_ata, u2_code_ata = np.loadtxt(path1 + '/Atlas/code_limiting_LDC_ata.dat', usecols = (1,2), unpack = True, dtype='float64')
u1_cla_ata, u2_cla_ata = np.loadtxt(path1 + '/Atlas/claret_limiting_LDC_ata.dat', usecols = (1,2), unpack = True, dtype='float64')

u1_code_pho, u2_code_pho = np.loadtxt(path1 + '/Phoenix/code_limiting_LDC_pho.dat', usecols = (1,2), unpack = True, dtype='float64')
u1_cla_pho, u2_cla_pho = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho.dat', usecols = (1,2), unpack = True, dtype='float64')
u1_cla_pho_r, u2_cla_pho_r = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho_r.dat', usecols = (1,2), unpack = True, dtype='float64')

u1, u1p, u1n, u2, u2p, u2n = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True, dtype='float64')

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

# For LDCs
# for name
name1 = []
for i in range(len(teff)):
	name1.append(name[i][:-1])

# CDS table
tmk = cdspyreadme.CDSTablesMaker()
tmk.title = "Empirical limb-darkening coefficients & transit parameters of known exoplanets from TESS"
tmk.author = "Jayshil A. Patel"
tmk.authors = "Jayshil A. Patel and Nestor Espinoza"

tab1 = Table()
tab1['name'], tab1['name'].info.format, tab1['name'].description = np.asarray(name1), '%s', 'Name of the host star'
tab1['u1_j'], tab1['u1_j'].info.format, tab1['u1_j'].description = u1, '%0.2f', 'u1 determined from juliet analysis'
tab1['u1_j_p'], tab1['u1_j_p'].info.format, tab1['u1_j_p'].description = u1p, '%1.2f', 'upper band of 68% credibility'
tab1['u1_j_n'], tab1['u1_j_n'].info.format, tab1['u1_j_n'].description = u1n, '%1.2f', 'lower band of 68% credibility'
tab1['u2_j'], tab1['u2_j'].info.format, tab1['u2_j'].description = u2, '%1.2f', 'u2 determined from juliet analysis'
tab1['u2_j_p'], tab1['u2_j_p'].info.format, tab1['u2_j_p'].description = u2p, '%1.2f', 'upper band of 68% credibility'
tab1['u2_j_n'], tab1['u2_j_n'].info.format, tab1['u2_j_n'].description = u2n, '%1.2f', 'lower band of 68% credibility'
tab1['u1_code_ata'], tab1['u1_code_ata'].info.format, tab1['u1_code_ata'].description = u1_code_ata, '%1.2f', 'u1 from EJ15 ATLAS'
tab1['u2_code_ata'], tab1['u2_code_ata'].info.format, tab1['u2_code_ata'].description = u2_code_ata, '%1.2f', 'u2 from EJ15 ATLAS'
tab1['u1_code_pho'], tab1['u1_code_pho'].info.format, tab1['u1_code_pho'].description = u1_code_pho, '%1.2f', 'u1 from EJ15 PHOENIX'
tab1['u2_code_pho'], tab1['u2_code_pho'].info.format, tab1['u2_code_pho'].description = u2_code_pho, '%1.2f', 'u2 from EJ15 PHOENIX'
tab1['u1_cla_ata'], tab1['u1_cla_ata'].info.format, tab1['u1_cla_ata'].description = u1_cla_ata, '%1.2f', 'u1 from C17 ATLAS'
tab1['u2_cla_ata'], tab1['u2_cla_ata'].info.format, tab1['u2_cla_ata'].description = u2_cla_ata, '%1.2f', 'u2 from C17 ATLAS'
tab1['u1_cla_pho'], tab1['u1_cla_pho'].info.format, tab1['u1_cla_pho'].description = u1_cla_pho, '%1.2f', 'u1 from C17 PHOENIX'
tab1['u2_cla_pho'], tab1['u2_cla_pho'].info.format, tab1['u2_cla_pho'].description = u2_cla_pho, '%1.2f', 'u2 from C17 PHOENIX'
tab1['u1_cla_pho_r'], tab1['u1_cla_pho_r'].info.format, tab1['u1_cla_pho_r'].description = u1_cla_pho_r, '%1.2f', 'u1 from C17 PHOENIX, with r-method'
tab1['u2_cla_pho_r'], tab1['u2_cla_pho_r'].info.format, tab1['u2_cla_pho_r'].description = u2_cla_pho_r, '%1.2f', 'u2 from C17 PHOENIX, with r-method'

"""
table = tmk.addTable(tab1, os.getcwd() + '/Tables/ldc_ascii_cds.dat', description=f"Limb darkening coefficients")
tmk.writeCDSTables()

templatename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','cdspyreadme','ReadMe.template')
tmk.setReadmeTemplate(templatename)#, templateValue)
with open("ReadMe", "w") as fd:
	tmk.makeReadMe(out=fd)
"""

tab1.write(os.getcwd() + '/Tables/ldc_ascii_mrt.dat', format='ascii.mrt', overwrite=True, delimiter='\t\t')#,\
	# formats={'name':'%20s', 'temp':'%4.1f', 'logg':'%1.3f', 'mh':'%1.2f', 'vturb':'%s'})