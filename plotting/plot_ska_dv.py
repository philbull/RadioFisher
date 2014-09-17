#!/usr/bin/python
"""
Plot dilation distance, D_V(z).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

names = ['cSKA1MIDfull1', 'cSKA1MIDfull2', 'SKA1SURfull1', 'SKA1SURfull2',
         'SKAHI100',] # 'SKAHI73', 'EuclidRef', 'LSST']
colours = ['#1619A1', '#1619A1', '#5B9C0A', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', '#19BCBC']
labels = ['SKA1-MID B1 IM', 'SKA1-MID B2 IM', 'SKA1-SUR B1 IM', 'SKA1-SUR B2 IM',
          'SKA1 HI gal.', 'SKA2 HI gal.', 'Euclid', 'LSST']
#linestyle = [[1,0], [8, 4], [1,0], [8, 4], [1,0], [1, 0], [2, 4, 6, 4], [8, 4]]
linestyle = [[], [], [], [], [], [], [], [], []]
#marker = ['o', 'D', 'o', 'D', 'o', 'D', 'o', 'D']
marker = [None, None, None, None, None, None, None, None, None, ]



names = ['SKAHI100', 'cSKA1MIDfull1', 'cSKA1MIDfull2', 'SKA1SURfull1', 'SKA1SURfull2', 'BOSS', 'SKAHI73'] # 'LSST']
colours = ['#5B9C0A', '#1619A1', '#8082FF', '#CC0000', '#FF8080',  '#990A9C', '#FFB928',   '#990A9C', '#FFB928', '#CC0000', '#19BCBC']
labels = ['SKA1 HI gal.', 'SKA1-MID B1 IM', 'SKA1-MID B2 IM', 'SKA1-SUR B1 IM', 'SKA1-SUR B2 IM', 'BOSS', 'SKA2 HI gal.', 'LSST']

names = ['EuclidRef', 'BOSS', 'SKAHI73', 'WFIRST', 'HETDEX'] # 'LSST']
colours = ['#5B9C0A', '#1619A1', '#8082FF', '#CC0000', '#FF8080',  '#990A9C', '#FFB928',   '#990A9C', '#FFB928', '#CC0000', '#19BCBC']
labels = ['Euclid Ref.', 'BOSS New', 'SKA2 HI gal.', 'WFIRST', 'HETDEX']

fname = 'ska-dv-5.png'



#########
names = [ 'fSKA1SURfull2_baoonly', 'SKA1MIDfull2_baoonly', 
          'gSKASURASKAP_baoonly', 'gSKAMIDMKB2_baoonly', 'gSKA2_baoonly',
          'BOSS_baoonly', 'EuclidRef_baoonly', 'WFIRST_baoonly', ] #'iMFAA']
labels = ['SKA1-SUR B2 (IM)', 'SKA1-MID B2 (IM)', 'SKA1-SUR (gal.)', 
          'SKA1-MID (gal.)', 'Full SKA (gal.)', 
          'BOSS (gal.)', 'Euclid (gal.)', 'WFIRST (gal.)', 'MFAA']
colours = ['#8082FF', '#1619A1', '#FFB928', '#ff6600', '#CC0000', 
           '#000000', '#858585', '#c1c1c1', 'y']          

          
names = [ 'gSKASURASKAP_baoonly', 'gSKAMIDMKB2_baoonly', 'gSKA2_baoonly',
          'BOSS_baoonly', 'EuclidRef_baoonly', ] #'iMFAA']
labels = ['SKA1-SUR (gal.)', 'SKA1-MID (gal.)', 'Full SKA (gal.)', 
          'BOSS (gal.)', 'Euclid (gal.)',]          
colours = ['#FFB928', '#ff6600', '#CC0000', 
           '#000000', '#858585', '#c1c1c1', 'y']
linestyle = [[], [], [], [], [], [], [], [], []]
marker = ['D', 'D', 's', 's', 's', 'o', 'o', 'o', 'o']
ms = [6., 6., 6., 6., 6., 5., 5., 5., 5.]
#########

"""
# FIXME
names = ["SKAHI100_BAOonly", "SKAHI73_BAOonly", 'EuclidRef', 'EuclidRef_BAOonly']
labels = ['SKA1 HI gal.', 'SKA2 HI gal.', 'Euclid', 'Euclid BAO-only']
colours = ['#1619A1', '#CC0000', '#FFB928', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
linestyle = [[1,0], [1, 0], [1, 0], [1, 0]]
marker = ['o', 'D', 's', 'o']
"""

# D_V(z) constraints from Font-Ribera et al. (arXiv:1308.4164)
z_boss = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
dv_boss = [0.0734, 0.0281, 0.0175, 0.0129, 0.0103, 0.0088, 0.0107, 0.0387]
z_eboss_elg = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15]
dv_eboss_elg = [0.2091, 0.0244, 0.0226, 0.0259, 0.064, 0.1339, 0.1367, 0.118, 0.1195, 0.1053, 0.106, 0.1145, 0.115, 0.1247, 0.1249, 0.1364, 0.1364]
z_eboss = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15]
dv_eboss = [0.1045, 0.013, 0.0166, 0.0244, 0.1851, 0.0669, 0.0683, 0.059, 0.0597, 0.0526, 0.053, 0.0572, 0.0575, 0.0624, 0.0625, 0.0682, 0.0682]
z_hetdex = [1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45]
dv_hetdex = [0.0349, 0.0348, 0.0347, 0.0346, 0.0346, 0.0345, 0.0345, 0.0344, 0.0344, 0.0343, 0.0343, 0.0343, 0.0343, 0.0343, 0.0342, 0.0342]
z_desi = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85]
dv_desi = [0.0195, 0.013, 0.01, 0.0082, 0.007, 0.006, 0.0053, 0.0052, 0.0058, 0.0059, 0.0058, 0.0058, 0.0064, 0.0074, 0.0093, 0.0145, 0.0223, 0.0298]
z_euclid = [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05]
dv_euclid = [0.0079, 0.0056, 0.005, 0.0048, 0.0046, 0.0046, 0.0046, 0.0047, 0.005, 0.0055, 0.0063, 0.0075, 0.0097, 0.0138, 0.0311]
z_wfirst = [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75]
dv_wfirst = [0.0103, 0.0098, 0.0092, 0.0088, 0.0085, 0.0084, 0.0084, 0.0086, 0.0088, 0.0093, 0.0157, 0.0162, 0.017, 0.0186, 0.0208, 0.0236, 0.0275, 0.0327]

# WiggleZ D_V constraints (1401.0358)
z_wigglez = [0.44, 0.60, 0.73]
dv_wigglez = [83./1716., 100./2221., 86./2516.]

#fr_z =  [z_boss, z_eboss_elg, z_eboss, z_hetdex, z_desi, z_euclid, z_wfirst]
#fr_dv = [dv_boss, dv_eboss_elg, dv_eboss, dv_hetdex, dv_desi, dv_euclid, dv_wfirst]
#fr_lbls = ['BOSS', 'eBOSS ELG', 'eBOSS', 'HETDEX', 'DESI', 'Euclid FR', 'WFIRST']


#P.plot(z_boss, dv_boss, 'k-', lw=2., label='frBOSS', alpha=0.5)
#P.plot(z_wigglez, dv_wigglez, 'y-', lw=2., label='WiggleZ', alpha=0.5)
#P.plot(z_wfirst, dv_wfirst, 'c-', lw=2., label='frWFIRST')
#P.plot(z_hetdex, dv_hetdex, 'm-', lw=2., label='frHETDEX')
#P.plot(z_euclid, dv_euclid, 'y-', lw=2., label='frEuclid')



# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    #pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
    #         'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #pnames += ["pk%d" % i for i in range(kc.size)]
    #zfns = [0,1,6,7,8]
    #excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    # Transform from D_A and H to D_V and F
    F_list_lss = []
    for i in range(Nbins):
        Fnew, pnames_new = rf.transform_to_lss_distances(
                              zc[i], F_list[i], pnames, DA=dAc[i], H=Hc[i], 
                              rescale_da=1e3, rescale_h=1e2)
        F_list_lss.append(Fnew)
    pnames = pnames_new
    F_list = F_list_lss
    
    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DV', 'F']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI',] #'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pDV = rf.indices_for_param_names(lbls, 'DV*')
    pFF = rf.indices_for_param_names(lbls, 'F*')
    
    DV = ((1.+zc)**2. * dAc**2. * C*zc / Hc)**(1./3.)
    Fz = (1.+zc) * dAc * Hc / C
    
    # Plot errors as fn. of redshift
    err = errs[pDV] / DV
    line = P.plot( zc, err, color=colours[k], lw=2.4, marker=marker[k], # lw=1.8
                   label=labels[k], markersize=ms[k], markeredgecolor=colours[k] )
    line[0].set_dashes(linestyle[k])
    
    print labels[k]
    for i in range(len(err)):
        print "%3.3f %4.4f" % (zc[i], err[i])
    
    #if 'Euclid' in names[k]:
    #    P.plot(z_desi, dv_desi, color='#FC80FF', lw=2., label='DESI')
    

# Plot Font-Ribera data
#for i in range(len(fr_z)):
#    P.plot(fr_z[i], fr_dv[i], 'k-', lw=2., label=fr_lbls[i])

# Subplot labels
P.gca().tick_params(axis='both', which='major', labelsize=18, width=1.5, size=8., pad=7.)
P.gca().tick_params(axis='both', which='minor', labelsize=18, width=1.5, size=8.)

# Set axis limits
##P.xlim((-0.05, 2.5))
####P.xlim((-0.02, 2.8))
P.xlim((-0.02, 2.2))
#P.ylim((0., 0.065))
##P.ylim((0., 0.045))
####P.ylim((0., 0.055))
P.ylim((0., 0.045))

P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel("$\sigma_{D_V}/D_V$", labelpad=15., fontdict={'fontsize':'xx-large'})
    
## Set tick locations
#ymajorLocator = matplotlib.ticker.MultipleLocator(0.02)
#yminorLocator = matplotlib.ticker.MultipleLocator(0.01)
P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.005))

#P.legend(loc='upper center', prop={'size':'medium'}, frameon=True, ncol=2)
leg = P.legend(loc='upper right', prop={'size':'large'}, frameon=False, ncol=2)
#frame = leg.get_frame()
#frame.set_edgecolor('w')

# Set size
P.gcf().set_size_inches(9.5, 6.8)
P.tight_layout()
P.savefig('ska-bao-dv-talk.png', transparent=True)
###P.savefig(fname, transparent=False)
#P.savefig('ska-dv.pdf', transparent=True)
#P.savefig('ska-dv-gal-bao-only.pdf', transparent=True)
#P.savefig('ska-dv-gal-bao-only-skaonly.pdf', transparent=True)
P.show()
