import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gd
import os
import pickle as pc
import utils1 as utl
from astropy.io import fits
from astroquery.mast import Observations as obs
from scipy import interpolate as inp
import matplotlib.cm as cm
import matplotlib.colors as cls
from scipy.optimize import curve_fit as cft
from pylab import *
import seaborn as sns


path1 = os.getcwd()#input('Enter the path of this folder: ')
#path1 = '/home/jayshil/Documents/Dissertation'
path2 = '/home/jayshil/Documents/Dissertation'

#os.system('mkdir ' + path1 + '/Light-curve')

os.system('mkdir ' + path1 + '/Results')
os.system('mkdir ' + path1 + '/Results/cal_us_and_evidance')
os.system('mkdir ' + path1 + '/Results/comp_a_r_p')
os.system('mkdir ' + path1 + '/Results/stellar_prop')
os.system('mkdir ' + path1 + '/Results/variation_with_temp')
#os.system('mkdir ' + path1 + '/limb-darkening-master/results')

#--------------------------------------------------------------------------------------------------
#--------------------------------Claret (2017) PHOENIX LDCs----------------------------------------
#--------------------------------------------------------------------------------------------------
f1 = open(path1 + '/Phoenix/claret_us_nl_pho.dat','w')#-------------------Non-linear-s-------------
f1.write('#Name\t\tc1\t\t\tc2\t\t\tc3\t\t\tc4\n')

f1r = open(path1 + '/Phoenix/claret_us_nl_pho_r.dat','w')#--------------------Non-linear-r------------
f1r.write('#Name\t\tc1\t\t\tc2\t\t\tc3\t\t\tc4\n')

f2 = open(path1 + '/Phoenix/claret_limiting_LDC_pho.dat','w')#-------------Limiting----------------
f2.write('#Name\t\tu1\t\tu2\n')

f2r = open(path1 + '/Phoenix/claret_limiting_LDC_pho_r.dat','w')#------------Limiting-r---------------
f2r.write('#Name\t\tu1\t\tu2\n')

#--------------------------------------------------------------------------------------------------
#--------------------------------Claret (2017) ATLAS LDCs------------------------------------------
#--------------------------------------------------------------------------------------------------
f3 = open(path1 + '/Atlas/claret_us_nl_ata.dat','w')#---------------------Non-linear---------------
f3.write('#Name\t\tc1\t\t\tc2\t\t\tc3\t\t\tc4\n')

f4 = open(path1 + '/Atlas/claret_limiting_LDC_ata.dat','w')##-------------Limiting------------------
f4.write('#Name\t\tu1\t\tu2\n')

#--------------------------------------------------------------------------------------------------
#----------------------------------Code ATLAS LDCs-------------------------------------------------
#--------------------------------------------------------------------------------------------------
f33 = open(path1 + '/Atlas/code_us_nl_ata.dat','w')#-------------------Non-linear------------------
f33.write('#Name\t\tc1\t\t\tc2\t\t\tc3\t\t\tc4\n')

f44 = open(path1 + '/Atlas/code_limiting_LDC_ata.dat','w')##-----------------Limiting---------------
f44.write('#Name\t\tu1\t\t\t\tu2\n')

#--------------------------------------------------------------------------------------------------
#-----------------------------------Code PHOENIX LDCs----------------------------------------------
#--------------------------------------------------------------------------------------------------
f11 = open(path1 + '/Phoenix/code_us_nl_pho.dat','w')#----------------------Non-linear-------------
f11.write('#Name\t\tc1\t\t\tc2\t\t\tc3\t\t\tc4\n')

f22 = open(path1 + '/Phoenix/code_limiting_LDC_pho.dat','w')#--------------Limiting----------------
f22.write('#Name\t\tu1\t\tu2\n')


#---------------------------------------------------------------------------------------------
#--------------------------Taking Data from the data file-------------------------------------
#---------------------------------------------------------------------------------------------

name2 = np.loadtxt('data_new.dat', dtype = str, usecols = 0, unpack = True)
teff2, lg2, mh2, p2, pperr2, tc2, aste2, asteperr2, rprst2, rprstperr2, tce2 =\
	 np.loadtxt('data_new.dat', usecols = (9,11,10,1,2,3,5,6,7,8,4), unpack = True)
vturb2 = np.ones(len(teff2))*2.0

#Name	Per	Per_err	Tc	Tc_err	a/R*	a/R*_err	Rp/R*	Rp/R*_err	Teff	MH	logg
#0		1	2		3	4		5		6			7		8			9		10	11

print('--------------------------------------------------------------------------------------------------------------------------')
print('----------------------------- Starting Third Iteration: To Calculate Theoretical LDCs-------------------------------------')
print('--------------------------------------------------------------------------------------------------------------------------')

for i in range(len(name2)):
	#--------------------------------------
	#------Making input files for----------
	#-------Limb Darkening Code------------
	#--------------------------------------
	fout = open(name2[i] + '.dat','w')
	fout.write('#Name\tTeff\tLog(g)\t[M/H]\tVturb\tRF\t\t\tFT\tmin_w\tmax_w\n')
	fout.write(name2[i] + '\t' + str(teff2[i]) + '\t' + str(lg2[i]) + '\t' + str(mh2[i]) + '\t' + str(vturb2[i])\
		 + '\ttess_res_fun.txt\tA100,P100\t-1\t-1')
	fout.close()
	utl.move_file(path2 + '/', name2[i] + '.dat', path2 + '/limb-darkening-master/input_files/', name2[i] + '.dat')
	#--------------------------------------
	#----Starting Limb-Darkening Code------
	#--------------------------------------
	os.system('python2.7 ' + path2 + '/limb-darkening-master/get_lds.py -ifile ' + path2 +'/limb-darkening-master/input_files/' +\
		 name2[i] + '.dat -ofile ' + name2[i] + '_LDC.dat')
	#--------------------------------------
	#---------------ATLAS------------------
	#--------------------------------------
	f5 = open(path2 + '/limb-darkening-master/results/' + name2[i] + '_LDC.dat', 'r')
	con = f5.readlines()
	line = con[24]
	ind = np.array([])
	for i1 in range(len(line)):
		if line[i1] == ',':
			ind = np.hstack((ind,i1))
	u1 = line[int(ind[3]-11):int(ind[3])]
	u2 = line[int(ind[4]-11):int(ind[4])]
	u3 = line[int(ind[5]-11):int(ind[5])]
	u4 = line[int(ind[5]+2):-1]
	f33.write(name2[i] + '\t\t' + u1 + '\t\t' + u2 + '\t\t' + u3 + '\t\t' + u4 + '\n')
	#--------------------------------------
	#-------------PHOENIX------------------
	#--------------------------------------
	line1 = con[36]
	ind1 = np.array([])
	for i2 in range(len(line1)):
		if line[i2] == ',':
			ind1 = np.hstack((ind1,i2))
	u1_p = line1[int(ind1[3]-11):int(ind1[3])]
	u2_p = line1[int(ind1[4]-11):int(ind1[4])]
	u3_p = line1[int(ind1[5]-11):int(ind1[5])]
	u4_p = line1[int(ind1[5]+2):-1]
	f11.write(name2[i] + '\t\t' + u1_p + '\t\t' + u2_p + '\t\t' + u3_p + '\t\t' + u4_p + '\n')
	#--------------------------------------
	#----Calculating LDCs from PHOENIX-q---
	#--------from Claret(2017)-------------
	#--------------------------------------
	lg1_p, T1_p, q1_p, q2_p, q3_p, q4_p = np.loadtxt(path1 + '/Phoenix/pho_ldc_claret.dat', usecols=(0,1,4,5,6,7), unpack=True)
	pts_p = np.vstack((lg1_p, T1_p))
	pt_p = np.transpose(pts_p)
	c1_p = inp.griddata(pt_p, q1_p, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c2_p = inp.griddata(pt_p, q2_p, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c3_p = inp.griddata(pt_p, q3_p, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c4_p = inp.griddata(pt_p, q4_p, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	f1.write(name2[i] + '\t' + str(c1_p) + '\t' + str(c2_p) + '\t' + str(c3_p) + '\t' + str(c4_p) + '\n')
	u1_p = ((12*c1_p)/35) + c2_p + ((164*c3_p)/105) + (2*c4_p)
	u2_p = ((10*c1_p)/21) - ((34*c3_p)/63) - c4_p
	f2.write(name2[i] + '\t' + str(u1_p) + '\t' + str(u2_p) + '\n')
	#--------------------------------------
	#----Calculating LDCs from PHOENIX-r---
	#--------from Claret(2017)-------------
	#--------------------------------------
	lg1_pr, T1_pr, q1_pr, q2_pr, q3_pr, q4_pr = np.loadtxt(path1 + '/Phoenix/pho_ldc_claret_r.dat', usecols=(0,1,4,5,6,7), unpack=True)
	pts_pr = np.vstack((lg1_pr, T1_pr))
	pt_pr = np.transpose(pts_pr)
	c1_pr = inp.griddata(pt_pr, q1_pr, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c2_pr = inp.griddata(pt_pr, q2_pr, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c3_pr = inp.griddata(pt_pr, q3_pr, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	c4_pr = inp.griddata(pt_pr, q4_pr, (lg2[i],teff2[i]), fill_value = 0, method = 'cubic')
	f1r.write(name2[i] + '\t' + str(c1_pr) + '\t' + str(c2_pr) + '\t' + str(c3_pr) + '\t' + str(c4_pr) + '\n')
	u1_pr = ((12*c1_pr)/35) + c2_pr + ((164*c3_pr)/105) + (2*c4_pr)
	u2_pr = ((10*c1_pr)/21) - ((34*c3_pr)/63) - c4_pr
	f2r.write(name2[i] + '\t' + str(u1_pr) + '\t' + str(u2_pr) + '\n')
	#--------------------------------------
	#-----Calculating LDCs from ATLAS------
	#--------from Claret(2017)-------------
	#--------------------------------------
	lg1_a, T1_a, met_a, q1_a, q2_a, q3_a, q4_a = np.loadtxt(path1 + '/Atlas/atlas_ldc_claret.dat', usecols=(0,1,2,4,5,6,7), unpack=True)
	pts_1 = np.vstack((lg1_a, T1_a))
	pts_a = np.vstack((pts_1,met_a))
	pt_a = np.transpose(pts_a)
	c1_a = inp.griddata(pt_a, q1_a, (lg2[i],teff2[i],mh2[i]), fill_value = 0, method = 'linear')
	c2_a = inp.griddata(pt_a, q2_a, (lg2[i],teff2[i],mh2[i]), fill_value = 0, method = 'linear')
	c3_a = inp.griddata(pt_a, q3_a, (lg2[i],teff2[i],mh2[i]), fill_value = 0, method = 'linear')
	c4_a = inp.griddata(pt_a, q4_a, (lg2[i],teff2[i],mh2[i]), fill_value = 0, method = 'linear')
	f3.write(name2[i] + '\t' + str(c1_a) + '\t' + str(c2_a) + '\t' + str(c3_a) + '\t' + str(c4_a) + '\n')
	u1_a = ((12*c1_a)/35) + c2_a + ((164*c3_a)/105) + (2*c4_a)
	u2_a = ((10*c1_a)/21) - ((34*c3_a)/63) - c4_a
	f4.write(name2[i] + '\t' + str(u1_a) + '\t' + str(u2_a) + '\n')
	print('****************************************************************************************')
	print('                                                                                        ')
	print('Calculated LDCs for ' + str(i+1) + ' system(s) / out of ' + str(len(name2)) + ' systems\n')
	print('                                                                                        ')
	print('****************************************************************************************')


f1.close()
f2.close()
f1r.close()
f2r.close()
f3.close()
f33.close()
f4.close()
f11.close()

#--------------------------------------------------------------------------------------------------
#----------------Calculating Limiting LDCs from ATLAS (Code)---------------------------------------
#--------------------------------------------------------------------------------------------------

name1 = np.loadtxt(path1 + '/Atlas/code_us_nl_ata.dat',dtype=str,usecols=0,unpack=True)
c1_code_a, c2_code_a, c3_code_a, c4_code_a = np.loadtxt(path1 + '/Atlas/code_us_nl_ata.dat', usecols=(1,2,3,4), unpack=True)

u1_code_a = ((12*c1_code_a)/35) + c2_code_a + ((164*c3_code_a)/105) + (2*c4_code_a)
u2_code_a = ((10*c1_code_a)/21) - ((34*c3_code_a)/63) - c4_code_a

for i in range(len(name1)):
	f44.write(name1[i] + '\t\t' + str(u1_code_a[i]) + '\t\t' + str(u2_code_a[i]) + '\n')

f44.close()

#--------------------------------------------------------------------------------------------------
#----------------Calculating Limiting LDCs from PHOENIX (Code)-------------------------------------
#--------------------------------------------------------------------------------------------------

name11 = np.loadtxt(path1 + '/Phoenix/code_us_nl_pho.dat',dtype=str,usecols=0,unpack=True)
c1_code_p, c2_code_p, c3_code_p, c4_code_p = np.loadtxt(path1 + '/Phoenix/code_us_nl_pho.dat', usecols=(1,2,3,4), unpack=True)

u1_code_p = ((12*c1_code_p)/35) + c2_code_p + ((164*c3_code_p)/105) + (2*c4_code_p)
u2_code_p = ((10*c1_code_p)/21) - ((34*c3_code_p)/63) - c4_code_p

for i in range(len(name11)):
	f22.write(name11[i] + '\t\t' + str(u1_code_p[i]) + '\t\t' + str(u2_code_p[i]) + '\n')

f22.close()

print("----------------------------------------------------------------------------------")
print("---------------------Your computing task is complete!!----------------------------")
print("----------------------------------------------------------------------------------")

print('----------------------------------------------------------------------------------')
print('------------------Now you can to plot some amazing results------------------------')
print('--------------------Please run plot.py to start plotting--------------------------')
print('----------------------------------------------------------------------------------')