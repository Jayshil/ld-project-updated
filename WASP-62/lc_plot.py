from pylab import *
import exotoolbox
import seaborn as sns
import pickle

def bin_data(x,y,n_bin):
    x_bins = []
    y_bins = []
    y_err_bins = []
    for i in range(0,len(x),n_bin):
        x_bins.append(np.median(x[i:i+n_bin-1]))
        y_bins.append(np.median(y[i:i+n_bin-1]))
        y_err_bins.append(np.sqrt(np.var(y[i:i+n_bin-1]))/np.sqrt(len(y[i:i+n_bin-1])))
    return np.array(x_bins),np.array(y_bins),np.array(y_err_bins)

sns.set_context("talk")
sns.set_style("ticks")

# Fonts:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size':12})
plt.rc('legend', **{'fontsize':7})

# Ticks to the outside:
rcParams['axes.linewidth'] = 1.2 
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'


fig, axs = plt.subplots(2, 1,gridspec_kw = {'height_ratios':[1,1]}, figsize=(12,12))
ax1 = axs[0]
ax2 = axs[1]
#fig = plt.figure(figsize=(12, 6)) 
#ax = fig.add_subplot(111)

sectors = ['WASP-126_sector0','WASP-126_sector10','WASP-126_sector11','WASP-126_sector12',\
           'WASP-126_sector13','WASP-126_sector14','WASP-126_sector5','WASP-126_sector6','WASP-126_sector7',\
           'WASP-126_sector8','WASP-126_sector9']

sectorname = [1,7,8,9,10,11,2,3,4,5,6]

P = 2.849375
#sectorname = [0,10,11,12,13,14,5,6,7,8,9]
parameter_name = 'r2_p1'
all_data = []
all_data_err = []
samples = np.zeros([len(sectors),2000])
sigmas = np.zeros(len(sectors))
for i in range(len(sectors)):
    phase,time,flux,flux_err,m = np.loadtxt(sectors[i]+'/results/exm1/phased_lc_planet1_TESS.dat',unpack=True)
    factor = (0.8/np.min(m-1.))
    m = (m-1.)*factor + sectorname[i]
    flux = (flux-1.)*factor + sectorname[i]
    flux_err = flux_err*factor
    idx = np.argsort(phase)
    xbin,ybin,ybin_err = bin_data(phase[idx],flux[idx],120)
    #print sectorname[i],'1hr error:',np,median(ybin_err)*1e6
    ax1.errorbar(flux,phase,xerr=flux_err,fmt='.',elinewidth=1,color='black',alpha=0.1,zorder=1)
    ax1.errorbar(ybin,xbin,xerr=ybin_err,fmt='o',markeredgewidth=1,ms=10,elinewidth=1,ecolor='black',mec='cornflowerblue',mfc='white',zorder=5)
    ax1.plot(m[idx],phase[idx],color='cornflowerblue',zorder=3)
    posteriors = pickle.load(open(sectors[i]+'/results/exm1/posteriors.pkl','r'))
    if parameter_name == 'r2_p1':
        posteriors['posterior_samples'][parameter_name] = (posteriors['posterior_samples'][parameter_name]**2)*1e6
    idx = np.random.choice(np.arange(len(posteriors['posterior_samples'][parameter_name])),2000,replace=False)
    samples[i,:] = posteriors['posterior_samples'][parameter_name][idx]
    data,data_err = np.median(posteriors['posterior_samples'][parameter_name]),\
                    np.sqrt(np.var(posteriors['posterior_samples'][parameter_name]))
    sigmas[i] = data_err
    all_data.append(data)
    all_data_err.append(data_err)
ax1.set_ylabel('Phase')
ax1.set_xlim([0,12])
ax1.set_ylim([-0.04,0.04])
ax1.set_xticks(range(1,12))
ax1.set_xticklabels([])
ax1.set_xlabel('(Amplified) Relative flux + Sector')
sectorname,all_data,all_data_err = np.array(sectorname),np.array(all_data),np.array(all_data_err)
print 'Errors in ppm and sum/sqrt(n)'
print sigmas
print np.sqrt(np.sum(sigmas**2))/np.double(len(sigmas))
samples = np.median(samples,axis=0)

pmean,vu,vd = exotoolbox.utils.get_quantiles(samples)
psigmau = vu-pmean
psigmad = pmean-vd
print pmean,psigmau,psigmad
print len(samples)

ax2.fill_between([0,12],[vd,vd],[vu,vu],color='grey',alpha=0.5)

if parameter_name == 'r2_p1':
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Transit depth (ppm)')
ax2.errorbar(np.array(sectorname),np.array(all_data),yerr=np.array(all_data_err),fmt='o',elinewidth=1,\
         ecolor='cornflowerblue',markeredgewidth=2,markeredgecolor='cornflowerblue',mfc='white',ms=12)

ax2.text(0.3,6280,'Combined $\delta = {0:d}^{{+{1:d}}}_{{-{2:d}}}$ ppm'.format(int(np.round(pmean)),int(np.round(psigmau)),int(np.round(psigmad))))
ax2.plot([0,12],[pmean,pmean],'--',color='black',linewidth=1)
x = range(1,12)
ax2.set_xlim([0,12])
ax2.set_xticks(x)
ax2.set_xticklabels(x)
plt.savefig('depth_plus_lc.pdf')
plt.savefig('depth_plus_lc.png',dpi=500)
