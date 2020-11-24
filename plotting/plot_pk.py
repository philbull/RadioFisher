#!/usr/bin/python
"""
Plot constraints on P(k) for an experiment, overlaying the errorbars over the 
fiducial power spectrum. (Fig. 29)
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import scipy.integrate
import scipy.interpolate

C = 3e5
cosmo = rf.experiments.cosmo
#names = ["SKA1MID350_nokfg_paper",]
names = ["iCosVis32x32_cvlim",]
colours = ['#1619A1', '#CC0000', '#5B9C0A', '#990A9C']

# Get f_bao(k) function
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

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
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = []; excl = []
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    print lbls
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Plot errorbars
    yup, ydn = rf.fix_log_plot(pk, cov*pk)
    
    # Fix for PDF
    #yup[np.where(yup > 1e1)] = 1e1
    #ydn[np.where(ydn > 1e1)] = 1e1
    ax.errorbar( kc, pk, yerr=[ydn, yup], color=colours[k], ls='none', 
                      lw=1.8, capthick=1.8, label=names[k], marker='.' )

# Plot fiducial power spectrum, P(k)
kk = np.logspace(-4., 0., 1000)
pk = cosmo['pk_nobao'](kk) * (1. + fbao(kk))
ax.plot(kk, pk, 'k-', lw=1.5)

"""
# Calculate horizon size
zeq = 3265. # From WMAP 9; horizon size is pretty insensitive to this
om = cosmo['omega_M_0']
ol = cosmo['omega_lambda_0']
h = cosmo['h']
orad = om / (1. + zeq)
aa = np.linspace(0., 1., 10000)
_z = 1./aa - 1.
integ = 1. / np.sqrt(om*aa + ol*aa**4. + orad)
rh = C / (100.*h) * scipy.integrate.cumtrapz(integ, aa, initial=0.)
k_hor = 2.*np.pi / rh
khor_z = scipy.interpolate.interp1d(_z[::-1], k_hor[::-1], kind='linear')

ax.axvline(khor_z(np.min(zc)), color='r', lw=5.)
ax.axvline(khor_z(np.max(zc)), color='r', lw=5.)
ax.axvline(khor_z(3.5), color='r', lw=5.)
"""

# Set limits
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((1.25e-3, 8e-2))
ax.set_ylim((9e3, 1.1e5))

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5, pad=8.)
ax.set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
ax.set_ylabel(r"$\mathrm{P}(k) \,[\mathrm{Mpc}^{3}]$", fontdict={'fontsize':'xx-large'})

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.,6.)
#P.savefig('fig29-pk-lowk.pdf', transparent=True)
P.show()
