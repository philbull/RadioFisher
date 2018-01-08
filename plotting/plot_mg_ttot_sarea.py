#!/usr/bin/python
"""
Plot parameter constraints as a function of t_tot and S_area
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os

cosmo = rf.experiments.cosmo

#DATAFILES = ["MGSCAN_1", "MGSCAN_0",]
#colours = ['#CC0000', '#FFB928']
#YMAX = 0.46
#enames = ['MID B1 Alt.', 'MID B1 Base']
#fname = 'mg-sarea-ttot-MIDB1.pdf'

#DATAFILES = ["MGSCAN_4", "MGSCAN_2",]
#colours = ['#CC0000', '#FFB928']
#colours = ['#1619A1', '#5DBEFF'] # '#007A10', '#1C7EC5'
#YMAX = 0.11
#enames = ['MID B2 Alt.', 'MID B2 Base']
#fname = 'mg-sarea-ttot-MIDB2.pdf'

#DATAFILES = ["MGSCAN_9", "MGSCAN_8",]
#colours = ['#FFB928', '#CC0000']
#YMAX = 0.28 #0.115
#enames = ['MID B1 Alt. + MeerKAT', 'MID B1 Rebase. + MeerKAT',]
#fname = 'mg-sarea-ttot-MIDB1.pdf'

DATAFILES = ["MGSCAN_11", "MGSCAN_10",]
colours = ['#5DBEFF', '#1619A1',]
YMAX = 0.115
enames = ['MID B2 Alt. + MeerKAT', 'MID B2 Rebase. + MeerKAT',]
fname = 'mg-sarea-ttot-MIDB2.pdf'


#colours = ['#a6a6a6', '#000000', '#5DBEFF', '#1C7EC5', '#1619A1', 
#           '#FFB928', '#ff6600', '#CC0000', '#95CD6D', 
#           '#007A10', '#ff6600',
#           '#858585', '#c1c1c1', 'c', 'm']

sareas = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
ttots = [1000, 3000, 5000, 10000]
dashes = [[2,2], [4,2,2,2], [4,4], []]

sigma_w0 = np.zeros((len(DATAFILES), len(ttots), len(sareas))) # N(t_tot) x N(S_area)
sigma_gam = np.zeros(sigma_w0.shape) # N(t_tot) x N(S_area)

exptnames = [None, None, None, None, None, None]
for i in range(len(DATAFILES)):
    # Load data from file
    f = open(DATAFILES[i], 'r')
    for line in f.readlines():
        try:
            ename, ttot, sarea, sig_w0, sig_gam = line.split(" ")
            exptnames[i] = ename
        except:
            print "\t\t", line
            continue
        sigma_w0[i, ttots.index(int(ttot)), sareas.index(int(sarea))] = sig_w0
        sigma_gam[i, ttots.index(int(ttot)), sareas.index(int(sarea))] = sig_gam
    f.close()

# Plot results
P.subplot(111)

for j in range(len(DATAFILES)):
    col = colours[j]
    for i in range(len(ttots)):
        #alpha = 1. - 0.8 * float(j) / float(len(DATAFILES))
        lbl = None
        if j == 1: lbl = "%d hrs" % ttots[i]
        P.plot(sareas, sigma_w0[j, i], lw=2.1, color=col, dashes=dashes[i],
               label=lbl )
        #       label="%s %d hrs" % (enames[j], ttots[i]) )
        print sigma_w0[j, i]

# color=colours[k], label=labels[k], lw=1.8,
#        marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=4.)

P.xlabel(r'$S_{\rm area}$ $[{\rm deg}^2]$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma(w_0)$', labelpad=15., fontdict={'fontsize':'xx-large'})

# Set tick locations
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1000))
#P.yscale('log')

# Legend
leg = P.legend(prop={'size':18}, loc='upper right', frameon=True, ncol=1)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.8)

P.ylim((0., YMAX))

print enames
print exptnames

# Experiment names
t1 = P.figtext(0.25, 0.88, enames[1], transform=P.gca().transAxes, fontsize=18, color=colours[1])
t2 = P.figtext(0.25, 0.82, enames[0], transform=P.gca().transAxes, fontsize=18, color=colours[0])

t1.set_bbox(dict(color='w', alpha=0.7))
t2.set_bbox(dict(color='w', alpha=0.7))

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
#P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
