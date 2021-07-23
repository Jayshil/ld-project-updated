from struct import unpack
from pylab import *
#import exotoolbox
import seaborn as sns
import pickle
import juliet
from matplotlib import rcParams

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

# Analysed data
dataset = juliet.load(input_folder='multisector_in_transit_ExpMatern')
res1 = dataset.fit(sampler='dynamic_dynesty')
# Making phases
P, t0 = np.median(res1.posteriors['posterior_samples']['P_p1']),\
        np.median(res1.posteriors['posterior_samples']['t0_p1'])

depth, depth_pe, depth_ne = np.loadtxt('depth.dat', usecols=(1,2,3), unpack=True)
#

sectors = ['TESS1', 'TESS2', 'TESS3', 'TESS4', 'TESS6', 'TESS7', 'TESS8', 'TESS9', 'TESS10', 'TESS11', 'TESS12', 'TESS13', 'TESS27', 'TESS28', 'TESS29', 'TESS30', 'TESS31', 'TESS32', 'TESS33', 'TESS34']
#['WASP-126_sector0','WASP-126_sector10','WASP-126_sector11','WASP-126_sector12',\
           #'WASP-126_sector13','WASP-126_sector14','WASP-126_sector5','WASP-126_sector6','WASP-126_sector7',\
           #'WASP-126_sector8','WASP-126_sector9']
#print(len(sectors))
#print(sectors)

sectorname = np.loadtxt('depth.dat', usecols=0, unpack=True)
sname = np.loadtxt('depth.dat', usecols=0, unpack=True, dtype=str)
sname2 = np.arange(1,21,1)
#print(sectorname)
#print(len(sectorname))

#sectorname = [0,10,11,12,13,14,5,6,7,8,9]
parameter_name = 'r2_p1'
all_data = []
all_data_perr = []
all_data_nerr = []
samples = np.zeros([len(sectors),2000])
sigmas = np.zeros(len(sectors))

for i in range(len(sectors)):
    phase = juliet.get_phases(dataset.times_lc[sectors[i]], P, t0)
    time,flux,flux_err = dataset.times_lc[sectors[i]], dataset.data_lc[sectors[i]], dataset.errors_lc[sectors[i]]
    m = res1.lc.model[sectors[i]]['deterministic']#np.loadtxt(sectors[i]+'/results/exm1/phased_lc_planet1_TESS.dat',unpack=True)
    factor = (0.8/np.min(m-1.))
    m = (m-1.)*factor + sname2[i]
    flux = (flux-1.)*factor + sname2[i]
    flux_err = flux_err*factor
    # For GP
    gp_model = res1.lc.model[sectors[i]]['GP']
    idx = np.argsort(phase)
    xbin,ybin,ybin_err = bin_data(phase[idx],flux[idx]-gp_model[idx],120)
    #print sectorname[i],'1hr error:',np,median(ybin_err)*1e6
    ax1.errorbar(flux-gp_model,phase,xerr=flux_err,fmt='.',elinewidth=1,color='black',alpha=0.1,zorder=1)
    ax1.errorbar(ybin,xbin,xerr=ybin_err,fmt='o',markeredgewidth=1,ms=10,elinewidth=1,ecolor='black',mec='cornflowerblue',mfc='white',zorder=5)
    ax1.plot(m[idx],phase[idx],color='cornflowerblue',zorder=3)
    #posteriors = pickle.load(open(sectors[i]+'/results/exm1/posteriors.pkl','r'))
    #if parameter_name == 'r2_p1':
    #    posteriors['posterior_samples'][parameter_name] = (posteriors['posterior_samples'][parameter_name]**2)*1e6
    #idx = np.random.choice(np.arange(len(posteriors['posterior_samples'][parameter_name])),2000,replace=False)
    #samples[i,:] = posteriors['posterior_samples'][parameter_name][idx]
    data,data_perr, data_nerr = depth[i], depth_pe[i], depth_ne[i]#np.median(posteriors['posterior_samples'][parameter_name]),\
                    #np.sqrt(np.var(posteriors['posterior_samples'][parameter_name]))
    sigmas[i] = data_perr
    all_data.append(data)
    all_data_perr.append(data_perr)
    all_data_nerr.append(data_nerr)

ax1.set_ylabel('Phase')
ax1.set_xlim([0,21])
ax1.set_ylim([-0.025,0.025])
ax1.set_xticks(range(1,21))
ax1.set_xticklabels([])
ax1.set_xlabel('(Amplified) Relative flux + Sector')
sectorname,all_data,all_data_perr, all_data_nerr = np.array(sectorname),np.array(all_data),np.array(all_data_perr), np.array(all_data_nerr)
print('Errors in ppm and sum/sqrt(n)')
print(sigmas)
print(np.sqrt(np.sum(sigmas**2))/np.double(len(sigmas)))
samples = ((res1.posteriors['posterior_samples']['p_p1'])**2)*1e6

pmean,vu,vd = juliet.utils.get_quantiles(samples)
psigmau = vu-pmean
psigmad = pmean-vd
print(pmean,psigmau,psigmad)
print(len(samples))

ax2.fill_between([0,21],[vd,vd],[vu,vu],color='grey',alpha=0.5)

if parameter_name == 'r2_p1':
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Transit depth (ppm)')
ax2.errorbar(np.array(sname2),np.array(all_data),yerr=[all_data_nerr, all_data_perr],fmt='o',elinewidth=1,\
         ecolor='cornflowerblue',markeredgewidth=2,markeredgecolor='cornflowerblue',mfc='white',ms=12)

ax2.text(0.3,12400,'Combined $\delta = {0:d}^{{+{1:d}}}_{{-{2:d}}}$ ppm'.format(int(np.round(pmean)),int(np.round(psigmau)),int(np.round(psigmad))))
ax2.plot([0,21],[pmean,pmean],'--',color='black',linewidth=1)
x = range(1,21)
ax2.set_xlim([0,21])
ax2.set_xticks(x)
ax2.set_xticklabels(sname)
plt.savefig('depth_plus_lc.pdf')
plt.savefig('depth_plus_lc.png',dpi=500)
