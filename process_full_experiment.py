#!/usr/bin/python
"""
Process Fisher matrices for a full experiment and output results in a 
plotting-friendly format.
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import os, sys

cosmo = experiments.cosmo

names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1mid", "SKA1MK", "aSKA1MK", "SKA1MK_alpha1", "SKA1MK_CV"]

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    k = 9 #11 # 7, 9, 10

print "="*50
print "Survey:", names[k]
print "="*50

# Root name for experiment to process
root = "output/" + names[k]


################################################################################
# Load data
################################################################################

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


################################################################################
# Get constraints on A(z), f(z), H(z), d_A(z)
################################################################################

Nbins = zc.size
F_base = 0; F_a = 0; F_b = 0; F_all = 0
F_gamma = 0; F_eos = 0

# Reshuffle F_list
#print "LENGTH:", len(F_list)
#xF = []
#for i in range(7, 7+Nbins): xF.append(F_list[i])
#F_list = xF


# EOS FISHER MATRIX
names = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
         'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
F_eos, lbls = baofisher.combined_fisher_matrix(F_eos_list, exclude=[2,7,8,  4,5,6,9,  12], expand=[1,], names=names) #1,6

# Add Planck priors
#F_eos[14, 14] += 1. / 0.012**2. # sigma_8 conservative
#F_eos[15, 15] += 1. / 0.0073**2. # n_s
#F_eos[28, 28] += 1. / 0.02**2. # omega_k
#F_eos[15, 15] += 1. / 0.02**2. # omega_DE
#F_eos[32, 32] += 1. / 0.01**2. # h

for i in range(len(lbls)):
    print "%2d %8s  %5.6f" % (i, lbls[i], 1. / np.sqrt(F_eos[i,i]))
print "*"*50

cov = np.linalg.inv(F_eos)
for i in range(len(lbls)):
    print "%2d %8s  %5.6f" % (i, lbls[i], np.sqrt(cov[i,i]))

print "Condition number:", np.linalg.cond(F_eos), F_eos.shape


"""
cov = F_eos[27:,27:] # dz=0.1
#cov = F_eos[84:,84:] # dz=0.03

#cov = np.outer(np.arange(5)+1., np.arange(5)+1.)
#cov[np.diag_indices(cov.shape[0])] *= 1.03

corr = np.zeros(cov.shape)
for i in range(cov.shape[0]):
    for j in range(cov.shape[0]):
        corr[i,j] = cov[i,j] / np.sqrt(cov[i,i] * cov[j,j])

print corr
"""
baofisher.plot_corrmat(F_eos, lbls)

baofisher.triangle_plot(np.zeros(F_eos.shape[0]), F_eos, lbls)

#baofisher.plot_corrmat(F_eos_list[0], names)



exit()
for i in range(Nbins):
    
    # Trim params we don't care about here
    # F: A(z), bHI(z), f(z), sig2, dA(z), H(z), [fNL], [omega_k_ng], [omega_DE_ng], pk
    # F_eos: A, bHI, f, sig2, omegak, omegaDE, w0, wa
    
    
    
    _F = baofisher.fisher_with_excluded_params(F_list[i], [9, 10, 11, 12])
    
    # EOS matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, omegak, omegaDE, w0, 
    #  wa, h, gamma)
    
    # Exclude (Tb, aperp, apar, gamma)
    # Expand bHI(z), f(z)
    _F = baofisher.fisher_with_excluded_params(F_eos_list[i], [2, 7, 8, 14])
    zfns_base = [4, 3, 2]
    FF = baofisher.fisher_with_excluded_params(_F, [3,]) #_F
    #FF = _F
    for idx in zfns_base:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_base += FF
    
    exit()
    
    
    # Expand fns. of z one-by-one for the current z bin. (Indices of fns. of z 
    # are given in reverse order, to make figuring out where they are in the 
    # updated matrix from the prev. step easier.)
    
    """
    ##########################################
    F_corr = np.zeros(_F.shape)
    for ii in range(_F.shape[0]):
        for jj in range(_F.shape[0]):
            F_corr[ii,jj] = _F[ii, jj] / np.sqrt(_F[ii,ii] * _F[jj,jj])
    P.matshow(F_corr, vmin=-1., vmax=1., cmap=matplotlib.cm.get_cmap("RdBu"))
    P.title("z = %3.3f" % zc[i])
    P.colorbar()
    
    locs, labels = P.xticks()
    labels = ['', 'A','bHI', 'f', 'sig2', 'dA', 'H']
    new_labels = [x.format(locs[ii]) for ii,x  in enumerate(labels)]
    P.xticks(locs, new_labels)
    locs, labels = P.yticks()
    P.yticks(locs, new_labels)
    
    P.xlim((-0.5, 5.5))
    P.ylim((5.5, -0.5))
    
    for tick in P.gca().xaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    
    P.savefig("corrcoeff-%s-%3.3f.png" % (names[k], zc[i]))
    #P.show()
    #exit()
    ##########################################
    """
    
    # f(z), dA(z), H(z)
    #zfns_base = [5, 4, 2]
    zfns_base = [4, 3, 2]
    FF = baofisher.fisher_with_excluded_params(_F, [3,]) #_F
    #FF = _F
    for idx in zfns_base:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_base += FF
    
    # Convert f(z) to a constraint on gamma
    # gamma, dA(z), H(z)
    #zfns_gamma = [5, 4]
    zfns_gamma = [4, 3]
    FF = _F.copy()
    FF = baofisher.fisher_with_excluded_params(_F, [3,]) # FIXME
    oma = baofisher.omegaM_z(zc[i], cosmo)
    df_dgamma = fc[i] * np.log(oma)
    FF[2,:] *= df_dgamma
    FF[:,2] *= df_dgamma
    for idx in zfns_gamma:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_gamma += FF
    
    # A(z), f(z), dA(z), H(z)
    zfns_a = [5, 4, 2, 0]
    FF = _F
    for idx in zfns_a:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_a += FF
    
    # b_HI(z), f(z), dA(z), H(z)
    zfns_b = [5, 4, 2, 1]
    FF = _F
    for idx in zfns_b:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_b += FF
    
    # A(z), b_HI(z), f(z), dA(z), H(z)
    zfns_all = [5, 4, 2, 1, 0]
    FF = _F
    for idx in zfns_all:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_all += FF
    
    # EOS: b_HI(z), f(z)
    zfns_eos = [2, ] #1]
    FF = F_eos_list[i]
    #FF = baofisher.fisher_with_excluded_params(FF, [3,]) # FIXME: Remove this
    for idx in zfns_eos:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_eos += FF


# Plot Fbase
FF = F_gamma
fiducial = np.zeros(FF.shape[0])
name = ["A", "bHI"]
#for i in range(Nbins): name.append("b%d"%i)
#for i in range(Nbins): name.append("f%d"%i)
name.append("gamma")
#name.append("sigma")
for i in range(Nbins): name.append("DA%d"%i)
for i in range(Nbins): name.append("H%d"%i)

print "F shape:", FF.shape, len(name)
#baofisher.triangle_plot(fiducial, FF, name)

#zc = np.ones(Nbins) # FIXME


################################################################################
# Invert matrices, extract fn(z) errorbars
################################################################################

# f(z), dA(z), H(z)
sig = np.sqrt( np.diag(np.linalg.inv(F_base)) )
sigma_base_f  = sig[baofisher.indexes_for_sampled_fns(2, zc.size, zfns_base)]
sigma_base_da = 1e3*sig[baofisher.indexes_for_sampled_fns(3, zc.size, zfns_base)] # 4
sigma_base_H  = 1e2*sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_base)] # 5

# gamma, dA(z), H(z)
sig = np.sqrt( np.diag(np.linalg.inv(F_gamma)) )
sigma_gamma = sig[baofisher.indexes_for_sampled_fns(2, zc.size, zfns_gamma)]
sigma_gamma_da = 1e3*sig[baofisher.indexes_for_sampled_fns(3, zc.size, zfns_gamma)] # 4
sigma_gamma_H  = 1e2*sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_gamma)] # 5
print "sigma(gamma):", sigma_gamma

P.plot(zc, sigma_base_da)
P.plot(zc, sigma_base_H)
P.show()

# A(z), f(z), dA(z), H(z)
sig = np.sqrt( np.diag(np.linalg.inv(F_a)) )
sigma_a_A  = sig[baofisher.indexes_for_sampled_fns(0, zc.size, zfns_a)]
sigma_a_f  = sig[baofisher.indexes_for_sampled_fns(2, zc.size, zfns_a)]
sigma_a_da = 1e3*sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_a)]
sigma_a_H  = 1e2*sig[baofisher.indexes_for_sampled_fns(5, zc.size, zfns_a)]

# b_HI(z), f(z), dA(z), H(z)
sig = np.sqrt( np.diag(np.linalg.inv(F_b)) )
sigma_b_b  = sig[baofisher.indexes_for_sampled_fns(1, zc.size, zfns_b)]
sigma_b_f  = sig[baofisher.indexes_for_sampled_fns(2, zc.size, zfns_b)]
sigma_b_da = 1e3*sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_b)]
sigma_b_H  = 1e2*sig[baofisher.indexes_for_sampled_fns(5, zc.size, zfns_b)]

# A(z), b_HI(z), f(z), dA(z), H(z)
sig = np.sqrt( np.diag(np.linalg.inv(F_all)) )
sigma_all_A  = sig[baofisher.indexes_for_sampled_fns(0, zc.size, zfns_all)]
sigma_all_b  = sig[baofisher.indexes_for_sampled_fns(1, zc.size, zfns_all)]
sigma_all_f  = sig[baofisher.indexes_for_sampled_fns(2, zc.size, zfns_all)]
sigma_all_da = 1e3*sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_all)]
sigma_all_H  = 1e2*sig[baofisher.indexes_for_sampled_fns(5, zc.size, zfns_all)]

#EOS: A, bHI(z), f(z), sig2, omegak, omegaDE, w0, wa
sig = np.sqrt( np.diag(np.linalg.inv(F_eos)) )
sigma_eos_A    = sig[baofisher.indexes_for_sampled_fns(0, zc.size, zfns_eos)]
sigma_eos_sig2 = sig[baofisher.indexes_for_sampled_fns(3, zc.size, zfns_eos)]
sigma_eos_ok   = sig[baofisher.indexes_for_sampled_fns(4, zc.size, zfns_eos)] #4
sigma_eos_de   = sig[baofisher.indexes_for_sampled_fns(5, zc.size, zfns_eos)] #5
sigma_eos_w0   = sig[baofisher.indexes_for_sampled_fns(6, zc.size, zfns_eos)] #6
sigma_eos_wa   = sig[baofisher.indexes_for_sampled_fns(7, zc.size, zfns_eos)] #7


################################################################################
# Print marginal errors for EOS
################################################################################

"""
P.subplot(111)
P.plot(zc, sigma_base_da/dAc, 'k-')
P.plot(zc, sigma_base_H/Hc, 'r-')
P.plot(zc, sigma_base_f/fc, 'b-')
P.ylim((0., 0.03))
P.title("%d" % k)
"""


print "-"*50
print names[k]
print "-"*50
print "sigma(A):          %f" % sigma_eos_A
print "sigma(sigma_NL^2): %f" % sigma_eos_sig2
print "sigma(omega_K):    %f" % sigma_eos_ok
print "sigma(omega_DE):   %f" % sigma_eos_de
print "sigma(w0):         %f" % sigma_eos_w0
print "sigma(wa):         %f" % sigma_eos_wa

#print "%f" % sigma_eos_A
#print "%f" % sigma_eos_sig2
#print "%f" % sigma_eos_ok
#print "%f" % sigma_eos_de
#print "%f" % sigma_eos_w0
#print "%f" % sigma_eos_wa

print "-"*50

#P.show()
exit()


################################################################################
# Extract P(k) errors
################################################################################

exit()
# Save data file
np.savetxt( "sigmaA-%d.dat"%k, np.column_stack((zc, sigma_all_A)) )
#exit()

P.subplot(111)
P.plot(zc, sigma_all_da/dAc, 'k-')
P.plot(zc, sigma_a_da/dAc, 'y--')
P.ylim((0., 1.))
P.show()

"""
P.subplot(111)
for j in range(0, zc.size, 1):
    #yup, ylow = baofisher.fix_log_plot(pk[j], pkerr[j]*pk[j])
    #P.errorbar(kc[j], pk[j], yerr=pk[j]*pkerr[j], marker='.')
    #P.errorbar(kc[j], pk[j], yerr=[ylow, yup], marker='.')
    P.plot(kc[j], pkerr[j], label="%3.2f"%zc[j], lw=1.5)

#P.ylim((1e1, 1e6))

#P.ylim((1e-3, 1e1))
#P.xlim((2e-3, 5e-1))

P.legend(loc='upper right', prop={'size':'x-small'})

P.xscale('log')
P.yscale('log')


P.show()
"""
#yup, ylow = baofisher.fix_log_plot(pkc * kcfac, pkerr*pkc*kcfac)
#P.errorbar(kc/0.7, pkc*kcfac, yerr=[ylow, yup], marker='.', ls='none', color='r')
