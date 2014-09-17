#!/usr/bin/python
"""
Make a triangle plot for a set of parameters.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI

import os
import euclid

print "Obsolete."
exit()

USE_DETF_PLANCK_PRIOR = True # If False, use Euclid prior instead
MARGINALISE_OVER_W0WA = True # Whether to fix or marginalise over (w0, wa)

cosmo = rf.experiments.cosmo

names = ['EuclidRef', 'cexptL', 'iexptM', 'iexptM'] #, 'exptS']
labels = ['DETF IV', 'Facility', 'Mature', 'Planck'] #, 'Snapshot']

# TESTING
names = ['cexptL', 'cexptL', 'cexptL']
labels = ['PPlanck', 'Planck', 'Facility']
#names = ['cexptL', 'iexptM', 'exptS']
#labels = ['Facility', 'Mature', 'Snapshot']
colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#FFB928', '#FFEA28'],
            ['#5B9C0A', '#BAE484'] ]

scale_idx = 2 #1 # Index of experiment to use as reference for setting the x,y scales
nsigma = 4. #4.2 # No. of sigma (of reference experiment 1D marginal) to plot out to

# Set-up triangle plot
Nparam = 6 # No. of parameters
fig = P.figure()
axes = [[fig.add_subplot(Nparam, Nparam, (j+1) + i*Nparam) for i in range(j, Nparam)] for j in range(Nparam)]

# Fixed width and height for each subplot
w = 1.0 / (Nparam+1.)
h = 1.0 / (Nparam+1.)
l0 = 0.1
b0 = 0.1

# Prepare to save 1D marginals
params_1d = []; params_lbls = []

# Loop though rf.experiments.
_k = range(len(names))[::-1] # Reverse order of rf.experiments.
for k in _k:
    root = "output/" + names[k]
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma'] #, 'Mnu']
    #if "Euclid" not in names[k]: pnames.append('Mnu')
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,  6,7,8,   14,] #15] # 4:sigma8
    #if "Euclid" not in names[k]: excl.append(15)
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Apply Planck prior
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    else:
        # Euclid Planck prior
        print "*** Using Euclid (Mukherjee) Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        Fe = euclid.planck_prior_full
        F_eucl = euclid.euclid_to_rf(Fe, cosmo)
        Fpl, lbls = rf.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
    
    # FIXME: Use Planck prior alone
    if labels[k] == 'PPlanck':
        lbls = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F2 = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo)
        Fpl = np.eye(F2.shape[0]+1) * 1e2 #sigma8
        Fpl[:F2.shape[0],:F2.shape[0]] = F2
    elif labels[k] == 'Planck':
        pass
    else:
        # Revert Fisher matrix to prev. values (without Planck prior)
        Fpl[:-1,:-1] = F
        tmp = Fpl[-1,-1]
        Fpl[-1,:] = 0.
        Fpl[:,-1] = 0.
        Fpl[-1,-1] = 1e2
    
    # Remove unwanted params
    fixed_params = ['w0', 'wa']
    Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=[lbls.index(p) for p in fixed_params] )
    
    # Invert matrices
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    params_1d.append(np.sqrt(np.diag(cov_pl)))
    params_lbls.append(lbls)
    
    # Set which parameters are going into the triangle plot
    params = ['h', 'omega_b', 'omegak', 'omegaDE', 'n_s', 'sigma8'][::-1]
    label = ['h', 'omega_b', 'omegak', 'omegaDE', 'n_s', 'sigma8'][::-1]
    fid = [ cosmo['h'], cosmo['omega_b_0'], 0., cosmo['omega_lambda_0'], cosmo['ns'], cosmo['sigma_8'] ][::-1]
    
    # Loop through rows, columns, repositioning plots
    # i is column, j is row
    for j in range(Nparam):
        for i in range(Nparam-j):
            ax = axes[j][i]
            
            # Hide tick labels for subplots that aren't on the main x,y axes
            if j != 0:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_visible(False)
            if i != 0:
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_visible(False)
            
            # Fiducial values
            ii = Nparam - i - 1
            x = fid[ii] #rf.experiments.cosmo['w0']
            y = fid[j] #rf.experiments.cosmo['wa']
            p1 = lbls.index(params[ii])
            p2 = lbls.index(params[j])
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            
            # Plot ellipse *or* 1D
            if p1 != p2:
                # Plot contours
                AAA = 1.
                if labels[k] == 'PPlanck': AAA = 0.5
                ww, hh, ang, alpha = rf.ellipse_for_fisher_params(
                                                      p1, p2, None, Finv=cov_pl)
                ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*ww, 
                            height=alpha[kk]*hh, angle=ang, fc=colours[k][kk], 
                            ec=colours[k][0], lw=1.5, alpha=0.5*AAA) for kk in [1,0]]
                for e in ellipses: ax.add_patch(e)
                
                # Centroid and axis scale
                if k == scale_idx:
                    sig1 = np.sqrt(cov_pl[p1,p1])
                    sig2 = np.sqrt(cov_pl[p2,p2])
                    ax.plot(x, y, 'kx')
                    ax.set_xlim((x-nsigma*sig1, x+nsigma*sig1))
                    ax.set_ylim((y-nsigma*sig2, y+nsigma*sig2))
            else:
                sig = np.sqrt(cov_pl[p1,p1])
                xx = np.linspace(x-20.*sig, x+20.*sig, 4000)
                yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-x)/sig)**2.)
                yy /= np.max(yy)
                ax.plot(xx, yy, ls='solid', color=colours[k][0], lw=1.5)
                
                # Match x scale, and hide y ticks
                if k == scale_idx:
                    ax.set_xlim((x-nsigma*sig, x+nsigma*sig))
                ax.tick_params(axis='y', which='both', left='off', right='off')
                ax.set_ylim((0., 1.08))
            
            # Set position of subplot
            pos = ax.get_position().get_points()
            ax.set_position([l0+w*i, b0+h*j, w, h])
            
            if j == 0:
                ax.set_xlabel(label[ii], fontdict={'fontsize':'20'}, labelpad=20.)
            #if i == Nparam-j-1: ax.set_title(label[ii], fontdict={'fontsize':'20'})
            if i == 0:
                ax.set_ylabel(label[j], fontdict={'fontsize':'20'}, labelpad=20.)
                ax.get_yaxis().set_label_coords(-0.3,0.5)


params = ['h', 'omega_b', 'omegaDE', 'n_s', 'sigma8']
print [names[kk] for kk in _k]
for p in params:
    idxs = [params_lbls[i].index(p) for i in range(3)]
    print "%9s: %5.5f %5.5f %5.5f" % (p, params_1d[0][idxs[0]], params_1d[1][idxs[1]], params_1d[2][idxs[2]])


# Set size and save
P.gcf().set_size_inches(16.5,10.5)
P.show()
