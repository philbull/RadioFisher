#!/usr/bin/python

import numpy as np
import pylab as P
import baofisher
from experiments import cosmo

def rho_c(z):
    """
    Critical density, in units of Msun Mpc^-2
    """
    E2 = cosmo['omega_M_0']*(1.+z)**3. + cosmo['omega_lambda_0']
    return 2.76e7 * (100.*cosmo['h'])**2. * E2

z = np.linspace(-0.1, 5., 1000)

oHI = baofisher.omega_HI(z, cosmo)
bHI = baofisher.bias_HI(z, cosmo)

# Load Prochaska & Wolfe data (10^8 Msun Mpc^-3)
pw_zmin, pw_zmax, pw_rhoHI, pw_errp, pw_errm = np.genfromtxt("HI_evolution/prochaska_wolfe_rhoHI.dat").T
pw_zc = 0.5 * (pw_zmin + pw_zmax)
pw_dz = 0.5 * (pw_zmax - pw_zmin)
pw_omegaHI = pw_rhoHI * 1e8 / rho_c(0.) / 1e-3
pw_err_omegaHI_p = pw_errp * 1e8 / rho_c(0.) / 1e-3
pw_err_omegaHI_m = pw_errm * 1e8 / rho_c(0.) / 1e-3

# Load Noterdaeme et al. data (omega_HI / 10^-3)
n_zmin, n_zmax, n_omegaHI, n_err = np.genfromtxt("HI_evolution/noterdaeme_omegaHI.dat").T
n_zc = 0.5 * (n_zmin + n_zmax)
n_dz = 0.5 * (n_zmax - n_zmin)

print "Critical density:", rho_c(0.), "Msun Mpc^-3"

# (1): omegaHI * b_HI * r = 0.43 x 10^-3 +/- 0.07 (stat.) +/- 0.4 (sys.)


"""
# Omega_HI x bias
P.subplot(311)
P.plot(z, oHI * bHI / 1e-3, 'k-', lw=1.5)
P.errorbar([0.8,], [0.43/0.9,], yerr=[0.07/0.9,], marker='.')
P.errorbar([0.8,], [0.43/0.95,], yerr=[0.07/0.95,], marker='.')
P.ylabel("$\Omega_\mathrm{HI} b_\mathrm{HI}$", fontdict={'fontsize':'x-large'})
P.xlabel("$z$", fontdict={'fontsize':'x-large'})
"""

# Omega_HI
P.subplot(111)

# Theory curves
P.plot(z, oHI / 1e-3, 'k-', lw=2.2)
P.plot(z, (0.45/0.65) * oHI / 1e-3, 'k--', lw=2.2)




# DLA: Meiring et al. (2011), http://arxiv.org/abs/arXiv:1102.3927
P.errorbar([0.17,], [1.4,], yerr=[[1.3,], [0.7,]], xerr=[0.17,], marker='v', ls='none', 
           markeredgecolor='none', color='#1619A1', lw=1.5, capthick=1.5, 
           markersize=8., label="COS / Meiring (2011)")

# DLA: Prochaska & Wolfe (2009)
P.errorbar(pw_zc, pw_omegaHI, xerr=pw_dz, yerr=[pw_err_omegaHI_p,-1.*pw_err_omegaHI_m],
           marker='s', ls='none', markeredgecolor='none', color='#1619A1', 
           lw=1.5, capthick=1.5, label="SDSS-DR5 / Prochaska (2009)")

# DLA: Noterdaeme et al., http://arxiv.org/abs/0908.1574v1
P.errorbar(n_zc, n_omegaHI, xerr=n_dz, yerr=n_err, marker='D', ls='none', 
           markeredgecolor='none', color='#1619A1', lw=1.5, capthick=1.5, 
           label="SDSS-DR7 / Noterdaeme (2009)")

# DLA: Rao et al. (2005), http://arxiv.org/abs/astro-ph/0509469v1
# FIXME: People seem to be using different definitions of Omega_HI(z)!
P.errorbar([0.609, 1.219], [0.97, 0.94], yerr=[0.36, 0.28], marker='^', ls='none', 
           markeredgecolor='none', color='#1619A1', markersize=8., 
           lw=1.5, capthick=1.5, label="HST / Rao (2005)")



# ALFALFA (Martin et al. 2010)
P.errorbar([0.,], [0.43,], [0.03,], marker='s', ls='none', markeredgecolor='none', color='#CC0000',  lw=1.5, capthick=1.5, label="ALFALFA / Martin (2010)")

# GMRT (Lah 2007), http://arxiv.org/abs/astro-ph/0701668
P.errorbar([0.24,], [0.91,], [0.42,], marker='v', ls='none', markeredgecolor='none', color='#CC0000', lw=1.5, capthick=1.5, markersize=8., label="GMRT / Lah (2007)")

# HIPASS (Zwaan 2005), http://arxiv.org/abs/astro-ph/0502257
P.errorbar([0.,], [0.35,], [0.04,], marker='^', ls='none', markeredgecolor='none', color='#CC0000', lw=1.5, capthick=1.5, markersize=8., label="HIPASS / Zwaan (2005)")



# Simulations
# Sim. (Khandai 2011), http://arxiv.org/abs/1012.1880 (N.B. new value in published vers.!)
P.errorbar([0.8,], [1.03,], [0.28,], marker='s', ls='none', markeredgecolor='none', color='#5B9C0A', lw=1.5, capthick=1.5, markersize=6., label="Sim. / Khandai (2011)")


# Zero line
P.axhline(0., ls='dotted', color='k', lw=1.5)

P.xlim((-0.08, 5.))
P.ylim((-0.19, 1.55))
P.ylabel("$\Omega_\mathrm{HI}(z) / 10^{-3}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
P.xlabel("$z$", fontdict={'fontsize':'xx-large'}, labelpad=15.)

P.legend(loc='lower right', prop={'size':11}, ncol=2)

# Bias
#P.subplot(212)
#P.plot(z, bHI, 'k-', lw=1.5)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)
#P.gca().set_position([0.15, 0.15, 0.8, 0.65])


# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig("pub-omegaHI-evol.pdf", transparent=True)
P.show()
