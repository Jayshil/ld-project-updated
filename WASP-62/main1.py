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

os.system('mkdir ' + path1 + '/Light-curve')


#---------------------------------------------------------------------------------------------
#--------------------------Taking Data from the data file-------------------------------------
#---------------------------------------------------------------------------------------------

name = np.loadtxt(path1 + '/data.dat', dtype = str, usecols = 0, unpack = True)
teff, lg, mh, vturb, p, pperr, pnerr, tc, aste, asteperr, astenerr, ecc, ome, rprst, rprstperr, rprstnerr, tce = np.loadtxt(path1 + '/data.dat', usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), unpack = True)

f_data = open('data2.dat','w')
f_data.write('#Name\tTeff\tlog(g)\t[M/H]\tVturb\tPeriod\tP+err\tP-err\tTc\ta/R*\ta+err\ta-err\tEccentricity\tOmega\tRp/R*\tr+err\tr-err\tTc-err\n')

print('------------------------------------------------------------------------------------------------------------------')
print('------------------------ Starting First Iteration: To Download data and analyse them------------------------------')
print('------------------------------------------------------------------------------------------------------------------')

for i in range(len(teff)):
	#--------------------------------------
	#--------Downloading Data--------------
	#--------------------------------------
	print('----------------------------------------------------------------')
	print('---------------Working on '+ name[i]+' -------')
	print('----------------------------------------------------------------')
	obt = obs.query_object(name[i],radius=0.001)
	b = np.array([])
	for j in range(len(obt['intentType'])):
		if obt['obs_collection'][j] == 'TESS' and obt['dataproduct_type'][j] == 'timeseries':
			b = np.hstack((b,j))
	if len(b) == 0:
		print('******************************************************************************************')
		print('                                                                                          ')
		print('Completed analysis for ' + str(i+1) + ' system(s) / out of ' + str(len(name)) + ' systems')
		print('                                                                                          ')
		print('******************************************************************************************')
		continue
	obsid1 = np.array([])
	for j1 in range(len(b)):
		ob = int(b[j1])
		obsid1 = np.hstack((obsid1,obt['obsid'][ob]))
	for i11 in range(len(obsid1)):
		print('-------------------------------------------------')
		print('--------Working on '+ name[i] + ' - Sector ' + str(i11) + ' -------')
		print('-------------------------------------------------')
		try:
			tab = obs.download_products(str(obsid1[i11]),extension='fits')
		except:
			continue
		for j2 in range(len(tab['Local Path'])):
			b1 = tab['Local Path'][j2]
			if b1[-7:] == 'lc.fits':
				c1 = j2
		try:
			d = tab['Local Path'][int(c1)]
		except:
			continue
		os.system('mv ' + path1 + d[1:] + ' ' + path1 + '/Light-curve/' + name[i] + '_sector' + str(i11) + '.fits')
		os.system('rm -r mastDownload')
		#--------------------------------------
		#-------Data Downloaded----------------
		#-----And stored at proper folder------
		#--------------------------------------
		#--------------------------------------
		#----Getting Data from fits file-------
		#--------------------------------------
		hdul = fits.open(path1 + '/Light-curve/' + name[i] + '_sector' + str(i11) + '.fits')
		h = hdul[1].data
		#--------------------------------------
		#------Saving Data to Numpy Array------
		#--------------------------------------
		time_np = np.array([])
		flux_np = np.array([])
		fluxerr_np = np.array([])
		for j3 in h['TIME']:
			time_np = np.hstack((time_np,j3))
		for j4 in h['PDCSAP_FLUX']:
			flux_np = np.hstack((flux_np,j4))
		for j5 in h['PDCSAP_FLUX_ERR']:
			fluxerr_np = np.hstack((fluxerr_np,j5))
		time_bjd = time_np + 2457000
		#--------------------------------------
		#--------Removing NaN values-----------
		#--------------------------------------
		nan_val_in_flux = np.isnan(flux_np)
		for j6 in range(len(flux_np)):
			j61 = np.bool(nan_val_in_flux[j6])
			if j61 is True:
				time_bjd[j6] = flux_np[j6]
				fluxerr_np[j6] = flux_np[j6]
			else:
				continue
		time_bjd = time_bjd[np.isfinite(time_bjd)]
		flux_np = flux_np[np.isfinite(flux_np)]
		fluxerr_np = fluxerr_np[np.isfinite(fluxerr_np)]
		#--------------------------------------
		#--------Calculating Relative flux-----
		#--And generating light-curve from it--
		#--------------------------------------
		median_flux = np.median(flux_np)
		rel_flux = flux_np/median_flux
		rel_flux_err = fluxerr_np/median_flux
		fig_lightcurve = plt.figure()
		plt.errorbar(time_bjd, rel_flux, yerr=rel_flux_err, fmt='.')
		plt.title('Light-curve for ' + name[i] + '_sector' + str(i11) + ' system')
		plt.xlabel('Time (in BJD)')
		plt.ylabel('Relative Flux')
		plt.grid()
		plt.savefig('Fig.png')
		plt.close(fig_lightcurve)
		utl.move_file(in_path = path1, fi_name='/Fig.png', out_path = path1 + '/Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/', new_name='Fig.png')
		#--------------------------------------
		#-------Creating Data file and---------
		#-------External Parameter file--------
		#--------------------------------------
		j7 = np.vstack((time_bjd, rel_flux))
		j77 = np.vstack((j7, rel_flux_err))
		data_to_be_dumped = np.transpose(j77)
		np.savetxt(name[i] + '_sector' + str(i11) +'_lc.dat', data_to_be_dumped, newline='\tTESS\n', delimiter='\t')
		utl.move_file(in_path = path1 + '/', fi_name = name[i] + '_sector' + str(i11) + '_lc.dat', out_path = path1 + '/Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/data/', new_name = name[i] + '_sector' + str(i11) + '_lc.dat')
		np.savetxt(name[i] + '_sector' + str(i11) + '_lceparam.dat', time_bjd, newline = '\tTESS\n', delimiter = '\t')
		utl.move_file(in_path = path1 + '/', fi_name = name[i] + '_sector' + str(i11) + '_lceparam.dat', out_path = path1 + '/Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) +'/data/', new_name = name[i] + '_sector' + str(i11) + '_lceparam.dat')
		#--------------------------------------
		#------Creating Prior File-------------
		#--------------------------------------
		phase = (time_bjd - tc[i])/p[i]
		phase1 = int(phase[0])+1
		for j8 in range(len(time_bjd)):
			phase2 = phase[j8]-phase1
			if phase2<0.0001:
				j88 = j8
		t0 = time_bjd[j88]
		f3_p = open(name[i] + '_sector' + str(i11) +'_priors_exm1.dat','w')#For Ex-M Kernel with 'a' uniformally distributed
		f3_p.write('#Physical parameters of the transiting system:\n')
		f3_p.write('P_p1\t\t\tNormal\t\t' + str(p[i]) + ',' + str(pperr[i]) + '\n')
		f3_p.write('r1_p1\t\t\tUniform\t\t0.0,1.0\n')
		f3_p.write('r2_p1\t\t\tUniform\t\t0.0,1.0\n')
		f3_p.write('a_p1\t\t\tUniform\t\t1,100\n')
		f3_p.write('t0_p1\t\t\tNormal\t\t' + str(t0) + ',0.1\n')
		f3_p.write('ecc_p1\t\t\tFIXED\t\t' + str(ecc[i]) + '\n')
		f3_p.write('omega_p1\t\tFIXED\t\t' + str(ome[i]) + '\n')
		f3_p.write('#Photometric priors for TESS photometry:\n')
		f3_p.write('q1_TESS\t\t\tUniform\t\t0.0,1.0\n')
		f3_p.write('q2_TESS\t\t\tUniform\t\t0.0,1.0\n')
		f3_p.write('mdilution_TESS\t\tFIXED\t\t1.0\n')
		f3_p.write('mflux_TESS\t\tNormal\t\t0.0,1.0\n')
		f3_p.write('sigma_w_TESS\t\tJeffreys\t0.1,10000\n')
		f3_p.write('GP_sigma_TESS\t\tJeffreys\t0.1,10000\n')
		f3_p.write('GP_rho_TESS\t\tUniform\t\t0.001,1e4\n')
		f3_p.write('GP_timescale_TESS\tUniform\t\t0.001,1e4')
		f3_p.close()
		utl.move_file(path1 + '/', name[i] + '_sector' + str(i11) + '_priors_exm1.dat', path1 + '/Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/priors/', name[i] + '_sector' + str(i11) + '_priors_exm1.dat')
		utl.new_dir(path1 + '/Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/results/exm1')
		#--------------------------------------
		#-------Created Prior Files------------
		#--------------------------------------
		#--------------------------------------
		#---------Running Juliet---------------
		#--------------------------------------
		#os.system('cd ~')
		#os.system('export LD_LIBRARY_PATH=/home/jayshil/MultiNest/lib/:$LD_LIBRARY_PATH~/.bashrc')
		#os.system('export PATH=$PATH:$HOME/.local/bin/~/.bashrc')
		#os.system('cd ' + path1)
		os.system('python juliet.py -lcfile Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/data/' + name[i] + '_sector' + str(i11) + '_lc.dat' + ' -lceparamfile Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/data/' + name[i] + '_sector' + str(i11) + '_lceparam.dat -ldlaw TESS-quadratic -lctimedef TESS-TDB -priorfile Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/priors/' + name[i] + '_sector' + str(i11) + '_priors_exm1.dat -ofolder Simulations/' + name[i] + '/' + name[i] + '_sector' + str(i11) + '/results/exm1 -nlive 500')#Ran simulation with Ex-M Kernel and making 'a' uniformally distributed
	f_data.write(name[i] + '\t' + str(teff[i]) + '\t' + str(lg[i]) + '\t' + str(mh[i]) + '\t' + str(vturb[i]) + '\t' + str(p[i]) + '\t' + str(pperr[i]) + '\t' + str(pnerr[i]) + '\t' + str(tc[i]) + '\t' + str(aste[i]) + '\t' + str(asteperr[i]) + '\t' + str(astenerr[i]) + '\t' + str(ecc[i]) + '\t' + str(ome[i]) + '\t' + str(rprst[i]) + '\t' + str(rprstperr[i]) + '\t' + str(rprstnerr[i]) + '\t' + str(tce[i]) + '\n')
	print('******************************************************************************************')
	print('                                                                                          ')
	print('Completed analysis for ' + str(i+1) + ' system(s) / out of ' + str(len(name)) + ' systems\n')
	print('                                                                                          ')
	print('******************************************************************************************')

f_data.close()

print('------------------------------------------------------------------------------------------------------------------')
print('--------------------------------- First Iteration Completed Successfully------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------')



print("----------------------------------------------------------------------------------")
print("---------------------Your computing task is complete!!----------------------------")
print("----------------------------------------------------------------------------------")