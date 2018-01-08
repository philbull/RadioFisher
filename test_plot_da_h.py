#!/usr/bin/python
"""
Plot functions of redshift.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import experiments_galaxy as egal
import os
import euclid
import scipy.integrate


cosmo = experiments.cosmo

names = ['gSKA2_baoonly',] # 'cexptL',]
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C', 'y'] # DETF/F/M/S
labels = ['SKA2 (Phil, BAO only)', 'Facility',]
linestyle = [[2, 4, 6, 4], [], [8, 4], [3, 4], [1,1]]


# Load Sahba's data
#dat = np.genfromtxt("SAHBA_output_Euclid_diff_14bins_S4.txt").T
dat = np.genfromtxt("Sahba_output_Fisher_bao_7_rms_s30k_3.txt").T
#dat = np.genfromtxt("Sahba_Fisher_bao_7_rms_s30k_2.txt").T
zz = dat[0]
logda = dat[1] / 1e2
loghh = dat[2] / 1e2
logff = dat[3] / 1e2

# Load Vsurvey, n(z) etc. output by Sahba
# z, Vsurvey, ngal, bias
#zs, Vs, ns, bs, beta = np.genfromtxt("SAHBA_test_all_terms_Euclid_4.txt").T
#zs, gz, bs, beta = np.genfromtxt("SAHBA_test_all_terms_SKA.txt").T
#zs, gz, bs, f = np.genfromtxt("SAHBA_test_all_terms_SKA2.txt").T

#zs, ns, vs, gs, fs, bs, beta = np.genfromtxt("SAHBA_test_all_terms_Euclid_5.txt").T
# z, ngal(h^3 Mpc^-3), Vsurvey(h^-3 Mpc^3), g(z) , beta*bias, bias, beta
zs, vs, dvdz2, dndz, ns, bs, beta, fs = np.genfromtxt("SAHBA_test_all_terms_SKA_7_rms_2.txt").T

cosmo_fns = baofisher.background_evolution_splines(cosmo)

# Calculate Vsurvey, ngal, bias for Euclid
#expt = egal.load_expt(egal.EuclidRef)
expt = egal.load_expt(egal.SKA2)
zc = 0.5 * (expt['zmin'] + expt['zmax'])

print "Sahba total: %3.3e" % np.sum(ns*vs)

bb = expt['b']

def vol(zmin, zmax):
    HH, rr, DD, ff = cosmo_fns
    _z = np.linspace(zmin, zmax, 1000)
    Vsurvey = C * scipy.integrate.simps(rr(_z)**2. / HH(_z), _z)
    Vsurvey *= 4. * np.pi
    return Vsurvey

zmin = zs - 0.5*(zs[1] - zs[0])
zmax = zs + 0.5*(zs[1] - zs[0])

h = 0.67
fsky = 30e3 / (4.*np.pi*(180./np.pi)**2.) # FIXME
V = fsky * np.array([vol(expt['zmin'][i], expt['zmax'][i]) for i in range(expt['zmin'].size)])
#V = fsky * h**3. * np.array([vol(zmin[i], zmax[i]) for i in range(zmax.size)])
VV = V * h**3.

"""
P.plot(zs, vs, 'k-', marker='.', lw=1.5, label='Sahba, Vsurvey')
P.errorbar(zs, V, xerr=0.1, marker='.', lw=1.2, ls='dashed', label='Phil, Vol, 30k deg^2, dz=0.2', color='r')
P.errorbar(zs, V2, xerr=0.05, marker='.', lw=1.2, ls='dashed', label='Phil, Vol, 30k deg^2, dz=0.1')
P.ylabel("Vol(z) [$h^{-3} Mpc^3$]")
P.xlabel("z")
P.legend(loc='upper left')
P.show()
exit()
"""


HH, rr, DD, ff = cosmo_fns

print "Ngal_Sahba = %3.3e" % np.sum(vs * ns)
print "Ngal_Phil  = %3.3e" % np.sum(V * expt['nz'])

print fsky

dD = 1./(1.+8.) / DD(8.)
gp = (DD(zc) * dD) * (1.+zc)
pbeta = ff(zs) / bs


"""
P.subplot(231)
#P.plot(zs, gs, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zc, gp, 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("g(z)")
P.legend(loc='upper left')

P.subplot(232)
P.plot(zs, fs, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zc, ff(zc), 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("f(z) = beta * bias(z)")

P.subplot(233)
P.plot(zs, vs, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zc, VV, 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("Vol(z)")

P.subplot(234)
P.plot(zs, bs, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zc, bb, 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("b(z)")

P.subplot(235)
P.plot(zs, vs*ns, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zc, V*expt['nz'], 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("n(z)*vol(z)")

P.subplot(236)
P.plot(zs, beta, 'k.', lw=1.5, ls='solid', label="Sahba")
P.plot(zs, pbeta, 'r.', lw=1.5, ls='dashed', label="Phil")
P.ylabel("beta(z)")

#print "zc     n_Sahba    n_Phil"
#for i in range(zs.size):
#    print "%2.2f:  %3.3e  %3.3e" % (zs[i], ns[i], expt['nz'][i]/0.7**3.)

#print np.max((VV*expt['nz']) / (vs*ns))
#print (0.7/h)**3.
P.show()

exit()
"""

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
    
    # FISHER MATRIX
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['H', 'DA', 'f',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'b_HI', 'aperp', 'apar', 'fs8', 
            'bs8', 'A', 'sigma_NL'] # Remove all but D_A and H
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    print F
    Nz = zc.size
    for j in range(Nz):
        print zc[j], F[j,j]*(dAc[j]**2.), F[j+Nz,j+Nz]*(Hc[j]**2.), dAc[j], Hc[j]
        #F[j,j+Nz]/np.sqrt(F[j,j]*F[j+Nz,j+Nz])
    print "-"*50
    
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    print lbls
    
    # Identify functions of z and get errors
    pDA = baofisher.indices_for_param_names(lbls, 'DA*')
    pH = baofisher.indices_for_param_names(lbls, 'H*')
    pf = baofisher.indices_for_param_names(lbls, 'f*')
    err_da = errs[pDA] / (dAc/1e3)
    err_h = errs[pH] / (Hc/1e2)
    err_f = errs[pf] / fc
        
    P.plot(zc, err_da, color=colours[k], lw=1.8, marker='o', 
           label="%s D_A(z)" % labels[k])
    P.plot(zc, err_h, color=colours[k], lw=1.8, marker='o', 
           label="%s H(z)" % labels[k], ls='dashed')
    P.plot(zc, err_f, color=colours[k], lw=1.8, marker='o', 
           label="%s f(z)" % labels[k], ls='dotted')

#fsky = 15e3 / (4.*np.pi*(180./np.pi)**2.)
fsky = 1.

P.plot(zz, logda/fsky, 'b-', lw=1.5, label="Sahba 7uJy D_A(z)")
P.plot(zz, loghh/fsky, 'b--', lw=1.5, label="Sahba 7uJy H(z)")
P.plot(zz, logff/fsky, 'b:', lw=1.5, label="Sahba 7uJy f(z)")

P.ylabel("$\sigma_Y/Y$")
P.xlabel("z")

# Legend
P.legend(loc='upper right', frameon=False)

# Set size
P.gcf().set_size_inches(10., 7.)
P.show()
