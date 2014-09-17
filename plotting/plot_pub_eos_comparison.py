#!/usr/bin/python
"""
Process EOS Fisher matrices and overplot results for several rf.experiments.
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

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1mid", "SKA1MK", "iSKA1MK", "aSKA1MK", "SKA1MK_A0"]
names = ["SKA1MK",] #["MeerKAT", "SKA1mid", "SKA1MK"]
#colours = ['#F9BA0F', '#F90F24', '#E068A9'] #'#D34F04']
colours = ['#22AD1A', '#3399FF', '#ED7624']

# Fiducial value and plotting
x = rf.experiments.cosmo['omega_lambda_0']; y = rf.experiments.cosmo['omega_k_0']
#alpha = [1.52, 2.48, 3.44]
fig = P.figure()
ax1 = fig.add_subplot(221) # ok - w0
ax2 = fig.add_subplot(222) # ok - wa
ax3 = fig.add_subplot(223) # w0 - wa
#ax4 = P.subplot(224)

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T

    # Load Fisher matrices and P(k) constraints as fn. of z
    F_list = []; F_eos_list = []
    kc = []; pk = []; pkerr = []
    for i in range(zc.size):
        F_list.append( np.genfromtxt(root+"-fisher-%d.dat" % i) )
        F_eos_list.append( np.genfromtxt(root+"-fisher-eos-%d.dat" % i) )
        _kc, _pk, _pkerr = np.genfromtxt(root+"-pk-%d.dat" % i).T
        kc.append(_kc); pk.append(_pk); pkerr.append(_pkerr)

    Nbins = zc.size
    
    # EOS FISHER MATRIX
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    zfns = [1,]
    F_eos, lbls = rf.combined_fisher_matrix( F_eos_list, 
                                                    exclude=[2,4,5,6,7,8,9,12], 
                                                    expand=zfns, names=pnames)
    # Overlay error ellipses as a fn. of z
    p1 = rf.indexes_for_sampled_fns(4, zc.size, zfns) # w0 # y
    #p2 = rf.indexes_for_sampled_fns(5, zc.size, zfns) # # x
    p3 = rf.indexes_for_sampled_fns(5, zc.size, zfns) # h
    p4 = rf.indexes_for_sampled_fns(6, zc.size, zfns) # gamma
    
    Finv = np.linalg.inv(F_eos) # Pre-invert, for efficiency
    i = 0
    
    ##################################
    # w0 - h
    ##################################
    x = rf.experiments.cosmo['w0']
    y = rf.experiments.cosmo['h']
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p1[i], p3[i], F_eos, Finv=Finv)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], lw=2., alpha=0.85) for kk in range(0, 2)]
    for e in ellipses: ax1.add_patch(e)
    if k == 0: ax1.plot(x, y, 'kx')
    
    ##################################
    # w0 - gamma
    ##################################
    x = rf.experiments.cosmo['w0']
    y = rf.experiments.cosmo['gamma']
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p1[i], p4[i], F_eos, Finv=Finv)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], lw=2., alpha=0.85) for kk in range(0, 2)]
    for e in ellipses: ax2.add_patch(e)
    if k == 0: ax2.plot(x, y, 'kx')
    cov = np.linalg.inv( [ [F_eos[p1[i],p1[i]], F_eos[p1[i],p4[i]]], [F_eos[p1[i],p4[i]], F_eos[p4[i],p4[i]]] ] )
    
    # Error ellipse for Euclid
    w, h, ang, alpha = rf.ellipse_for_fisher_params(0, 1, euclid.cov_gamma_w_ref, Finv=euclid.cov_gamma_w_ref)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='k', lw=2., alpha=0.85) for kk in range(0, 2)]
    for e in ellipses: ax2.add_patch(e)
    if k == 0: ax2.plot(x, y, 'kx')
    
    """
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    majorLocator   = MultipleLocator(0.01)
    majorFormatter = FormatStrFormatter('%3.2f')
    minorLocator   = MultipleLocator(0.001)
    ax2.xaxis.set_major_locator(majorLocator)
    #ax2.xaxis.set_minor_locator(minorLocator)
    ax2.xaxis.set_major_formatter(majorFormatter)
    
    majorLocator   = MultipleLocator(0.01)
    majorFormatter = FormatStrFormatter('%3.2f')
    minorLocator   = MultipleLocator(0.001)
    ax2.yaxis.set_major_locator(majorLocator)
    ax2.yaxis.set_minor_locator(minorLocator)
    ax2.yaxis.set_major_formatter(majorFormatter)
    
    ax2.grid(True, which='both')
    
    ax2.axvline(x - sig_x)
    ax2.axvline(x + sig_x)
    ax2.axhline(y - sig_y)
    ax2.axhline(y + sig_y)
    """
    ax2.grid(True, which='both')
    
    ##################################
    # h - gamma
    ##################################
    x = rf.experiments.cosmo['h']
    y = rf.experiments.cosmo['gamma']
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p3[i], p4[i], F_eos, Finv=Finv)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], lw=2., alpha=0.85) for kk in range(0, 2)]
    for e in ellipses: ax3.add_patch(e)
    if k == 0: ax3.plot(x, y, 'kx')


lines = []
for i in range(len(names)):
    print i
    l = matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[i])
    lines.append(l)

fig.legend((l for l in lines), (name for name in names), (0.6, 0.25), prop={'size':'xx-large'})


ax1.set_xlabel(r"$w_0$", fontdict={'fontsize':'20'})
ax1.set_ylabel(r"$h$", fontdict={'fontsize':'20'})

ax2.set_xlabel(r"$w_0$", fontdict={'fontsize':'20'})
ax2.set_ylabel(r"$\gamma$", fontdict={'fontsize':'20'})

ax3.set_xlabel(r"$h$", fontdict={'fontsize':'20'})
ax3.set_ylabel(r"$\gamma$", fontdict={'fontsize':'20'})

axs = [ax1, ax2, ax3]
fontsize = 16.
for ax in axs:
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)



################################################################################

root = "/home/phil/oslo/de_function_priors/mcmc/"
fname_planck = "base_w_wa_planck_lowl_lowLike_SNLS_post_lensing_1.txt"

NBINS = 20
def get_centroids(vals):
	"""Get bin centroids"""
	cent = []
	for i in range(len(vals)-1):
		cent.append(0.5*(vals[i]+vals[i+1]))
	return cent


def get_sig_levels(z):
	"""
	Get values corresponding to the different significance levels for a 
	histo - argument is the histogrammed data
	"""
	linz = z.flatten()
	linz = np.sort(linz)[::-1]
	tot = sum(linz)
	acc = 0.0
	i=-1
	j=0
	#lvls = [0.0, 0.68, 0.95, 0.997, 1.0, 1.01] # Significance levels (Gaussian)
	lvls = [0.0, 0.68, 0.95, 1.0, 1.01] # Significance levels (Gaussian)
	slevels = []
	for item in linz:
		acc += item/tot
		i+=1
		if(acc >= lvls[j]):
			print "Reached " + str(j) + "-sigma at", item, "-- index", i
			j+=1
			slevels.append(item)
	slevels.append(-1e-15) # Very small number at the bottom
	return slevels[::-1]

 
# Load Planck MCMC chain
dat = np.genfromtxt(root+fname_planck).T

planck_w0 = dat[6]
planck_wa = dat[7]
planck_ol = dat[26]
planck_om = dat[27]
planck_ok = 1. - planck_ol - planck_om # Flatness is imposed!

# Plot Planck contours
#f, x, y = np.histogram2d(planck_w0, planck_wa, bins=NBINS)
#ax3.contour(get_centroids(x), get_centroids(y), f.T, get_sig_levels(f), linewidths=2.5)


P.show()
