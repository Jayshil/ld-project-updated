import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gd
from matplotlib import rcParams
import os
import pickle as pc
#import utils1 as utl
from astropy.io import fits
from astroquery.mast import Observations as obs
from scipy import interpolate as inp
import matplotlib.cm as cm
import matplotlib.colors as cls
from scipy.optimize import curve_fit as cft
from pylab import *
import seaborn as sns
from uncertainties import ufloat

sns.set_context("talk")
sns.set_style("ticks")

# Fonts:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size':12})
plt.rc('legend', **{'fontsize':12})

# Ticks to the outside:
rcParams['axes.linewidth'] = 1.2 
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'


# Let's first gather all the data
# Tables
u1_C17_phoq_t, u2_C17_phoq_t = ufloat(-0.13254920511822632, 0.01779116409692707), ufloat(0.2429583977944228, 0.021901472100136297)
u1_C17_phor_t, u2_C17_phor_t = ufloat(-0.0724350755763925, 0.017361956456837687), ufloat(0.10726822221692152, 0.021707774667636025)
u1_C17_ata_t, u2_C17_ata_t = ufloat(-0.10063315138893629, 0.017628666042063423), ufloat(0.12820975806256674, 0.021677122752321767)
u1_EJ15_pho_t, u2_EJ15_pho_t = ufloat(-0.05804935418751635, 0.017617176703639225), ufloat(0.10671418468101124, 0.02151903768425912)
u1_EJ15_ata_t, u2_EJ15_ata_t = ufloat(-0.0965586007189337, 0.017943376753683363), ufloat(0.14521764159664297, 0.021714707218830717)

u1_t, u2_t = np.array([u1_EJ15_ata_t, u1_EJ15_pho_t, u1_C17_ata_t, u1_C17_phoq_t, u1_C17_phor_t]),\
             np.array([u2_EJ15_ata_t, u2_EJ15_pho_t, u2_C17_ata_t, u2_C17_phoq_t, u2_C17_phor_t])

# SPAM
u1_C17_phoq_s, u2_C17_phoq_s = ufloat(-0.03165192484287202, 0.017381785845619634), ufloat(0.06815541169966002, 0.022020340192295797)
u1_C17_phor_s, u2_C17_phor_s = ufloat(-0.017852237016612836, 0.0175116877879471), ufloat(0.04047059301220022, 0.021911959798690532)
u1_C17_ata_s, u2_C17_ata_s = ufloat(-0.08150097042082566, 0.01766020078007795), ufloat(0.09188807962860449, 0.02158255712786697)
u1_EJ15_pho_s, u2_EJ15_pho_s = ufloat(-0.016779590764658386, 0.017618332262775364), ufloat(0.04216044785272996, 0.02180526185046541)
u1_EJ15_ata_s, u2_EJ15_ata_s = ufloat(-0.072985980098237, 0.01754948975750842), ufloat(0.09898236267098662, 0.02170070324362767)

u1_s, u2_s = np.array([u1_EJ15_ata_s, u1_EJ15_pho_s, u1_C17_ata_s, u1_C17_phoq_s, u1_C17_phor_s]),\
             np.array([u2_EJ15_ata_s, u2_EJ15_pho_s, u2_C17_ata_s, u2_C17_phoq_s, u2_C17_phor_s])

# For u1
# set width of bar
barWidth = 0.25
#fig = plt.subplots(figsize =(12, 8))
fig = plt.figure(figsize = (12,8))
gs1 = gd.GridSpec(2, 1, height_ratios = [1,1])

ax2 = plt.subplot(gs1[1])
ax1 = plt.subplot(gs1[0])#, sharex = ax2)
#ax2 = plt.subplot(gs1[1], sharex = ax1)

# Set position of bar on X axis
br1 = np.arange(len(u1_s))
br2 = [x + barWidth for x in br1]
 
# Make the plot (Bottom Panel)
for i in range(len(u1_s)):
    if i == 0:
        ax2.bar(br1[i], u1_t[i].n, color ='orangered', width = barWidth,
                edgecolor ='orangered', alpha=0.5, label ='Values from LDC Tables/Codes')
        ax2.bar(br2[i], u1_s[i].n, color ='cornflowerblue', width = barWidth,
                edgecolor ='cornflowerblue', alpha=0.5, label ='SPAM LDCs')
    else:
        ax2.bar(br1[i], u1_t[i].n, color ='orangered', width = barWidth,
                edgecolor ='orangered', alpha=0.5)
        ax2.bar(br2[i], u1_s[i].n, color ='cornflowerblue', width = barWidth,
                edgecolor ='cornflowerblue', alpha=0.5)
    ax2.errorbar(br1[i], u1_t[i].n, yerr=u1_t[i].s, fmt='.', c='orangered', mfc='white', mec='orangered', mew=2, ms=8)
    ax2.errorbar(br2[i], u1_s[i].n, yerr=u1_s[i].s, fmt='.', c='cornflowerblue', mfc='white', mec='cornflowerblue', mew=2, ms=8)

ax2.set_ylim([-0.27, 0.0])
ax2.set_ylabel(r'$\Delta u_1$', fontsize = 18)
ax2.xaxis.tick_top()
ax2.set_xticks([r + (barWidth/2) for r in range(len(u1_s))])
ax2.set_xticklabels([])
ax2.text(-0.25,-0.23,r'Mean offset ($\Delta u_1$):', fontsize=18)
ax2.text(-0.25,-0.26,r'Theoretical LDCs $-$ Empirical (TESS) LDCs', fontsize=18)
ax2.legend(loc='lower right', fontsize=15)

#ax2.ylabel('Mean Offset', fontweight ='bold', fontsize = 15)

# Make the plot (top panel)
for i in range(len(u2_s)):
    if i == 0:
        ax1.bar(br1[i], u2_t[i].n, color ='orangered', width = barWidth,
                edgecolor ='orangered', alpha=0.5, label ='Values from LDCs Tables/Codes')
        ax1.bar(br2[i], u2_s[i].n, color ='cornflowerblue', width = barWidth,
                edgecolor ='cornflowerblue', alpha=0.5, label ='SPAM LDCs')
    else:
        ax1.bar(br1[i], u2_t[i].n, color ='orangered', width = barWidth,
                edgecolor ='orangered', alpha=0.5)
        ax1.bar(br2[i], u2_s[i].n, color ='cornflowerblue', width = barWidth,
                edgecolor ='cornflowerblue', alpha=0.5)
    ax1.errorbar(br1[i], u2_t[i].n, yerr=u1_t[i].s, fmt='.', c='orangered', mfc='white', mec='orangered', mew=2, ms=8)
    ax1.errorbar(br2[i], u2_s[i].n, yerr=u1_s[i].s, fmt='.', c='cornflowerblue', mfc='white', mec='cornflowerblue', mew=2, ms=8)

#ax1.ylabel('Mean Offset', fontweight ='bold', fontsize = 15)
ax1.set_ylim([0.0, 0.27])
ax1.set_xticks([r + (barWidth/2) for r in range(len(u1_s))])
#ax1.set_xticklabels([r'EJ15 (\textsc{atlas})', r'EJ15 (\textsc{phoenix})', r'C17 (\textsc{atlas})',\
#         r'C17 (\textsc{phoenix}:\\ \textit{q-method})', r'C17 (\textsc{phoenix}:\\ \textit{r-method})'], rotation=0, fontsize=15)
ax1.set_xticklabels([r'EJ15 (\textsc{atlas})', r'EJ15 (\textsc{phoenix})', r'C17 (\textsc{atlas})',\
         r'C17q (\textsc{phoenix})', r'C17r (\textsc{phoenix})'], rotation=0, fontsize=15)
ax1.set_ylabel(r'$\Delta u_2$', fontsize = 18)
ax1.text(-0.25,0.24,r'Mean offset (for $\Delta u_2$):',fontsize=18)
ax1.text(-0.25,0.21,r'Theoretical LDCs $-$ Empirical (TESS) LDCs', fontsize=18)

# Adding Xticks
#plt.xlabel('', fontweight ='bold', fontsize = 15)

#plt.ylabel('Mean Offset', fontweight ='bold', fontsize = 15)
plt.subplots_adjust(hspace = 0.25)

#plt.xticks([r + (barWidth/2) for r in range(len(u1_s))],
#        ['EJ15 (Atlas)', 'EJ15 (Phoenix)', 'C17 (Claret)', 'C17 (Phoenix - q)', 'C17 (Phoenix - r)'])
#plt.tight_layout()
#plt.legend()
plt.savefig(os.getcwd() + '/Mean-off/mean_off.pdf')
plt.show()