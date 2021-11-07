import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat


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
fig = plt.subplots(figsize =(12, 6.75))

# Set position of bar on X axis
br1 = np.arange(len(u1_s))
br2 = [x + barWidth for x in br1]
 
# Make the plot
for i in range(len(u1_s)):
    if i == 0:
        plt.bar(br1[i], np.abs(u1_t[i].n), color ='r', width = barWidth,
                edgecolor ='grey', alpha=0.5, label ='Tables')
        plt.bar(br2[i], np.abs(u1_s[i].n), color ='g', width = barWidth,
                edgecolor ='grey', alpha=0.5, label ='SPAM')
    else:
        plt.bar(br1[i], np.abs(u1_t[i].n), color ='r', width = barWidth,
                edgecolor ='grey', alpha=0.5)
        plt.bar(br2[i], np.abs(u1_s[i].n), color ='g', width = barWidth,
                edgecolor ='grey', alpha=0.5)
    plt.errorbar(br1[i], np.abs(u1_t[i].n), yerr=np.abs(u1_t[i].s), fmt='.', c='r')
    plt.errorbar(br2[i], np.abs(u1_s[i].n), yerr=np.abs(u1_s[i].s), fmt='.', c='g')
 
# Adding Xticks
#plt.xlabel('', fontweight ='bold', fontsize = 15)
plt.ylabel('Mean Offset', fontweight ='bold', fontsize = 15)
plt.xticks([r + (barWidth/2) for r in range(len(u1_s))],
        ['EJ15 (Atlas)', 'EJ15 (Phoenix)', 'C17 (Claret)', 'C17 (Phoenix - q)', 'C17 (Phoenix - r)'])
 
plt.legend()
plt.show()