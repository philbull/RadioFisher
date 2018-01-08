#!/usr/bin/python
"""
Plot relative noise factors for different SKA telescopes.
"""
import numpy as np
import pylab as P
import radiofisher as rf

e = rf.experiments

def noise_factor(z, expt, paf=False):
    """
    Calculate the autocorrelation noise factor for a given redshift.
    """
    z = np.array(z)
    nu = expt['nu_line'] / (1.+z)
    noise = np.ones(z.shape)
    
    # T_sys
    Tsky = 60e3 * (300.*(1. + z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky # System temperature
    noise *= Tsys**2.
    
    # PAF beam correction
    if paf:
        idx = np.where(nu <= expt['nu_crit'])
        noise[idx] *= (expt['nu_crit'] / nu[idx])**2.
    
    # Dish/beam factor
    noise *= 1. / (expt['Ndish'] * expt['Nbeam'])
    
    # Zero outside of specified redshift range
    numax = expt['survey_numax']
    numin = numax - expt['survey_dnutot']
    idx = np.where(np.logical_or(nu > numax, nu < numin))
    noise[idx] = 0.
    return noise

z = np.linspace(0., 3., 1000)

expts = [e.SKA0MID, e.SKA1MID900, e.SKA1MID350, e.SKA0SUR, e.SKA1SUR650, e.SKA1SUR350]
lbl = ['SKA0-MID', 'SKA1-MID 900', 'SKA1-MID 350', 
       'SKA0-SUR', 'SKA1-SUR 650', 'SKA1-SUR 350']
paf = [False, False, False, True, True, True]
col = ['#FFD233', '#FF8333', '#a82222', '#7BCDE5', '#4891E8', '#4348C0']

# Normalise to SKA1-MID 900 at z=0
norm = noise_factor([0.01,], e.SKA1MID900, paf=False)

# Plot relative noise factor
P.subplot(111)
for i in range(len(expts)):
    n = noise_factor(z, expts[i], paf=paf[i])
    P.plot(z, n / norm, label=lbl[i], lw=2.2, color=col[i])

P.axhline(1., lw=1.5, ls='dotted', color='k')
P.yscale('log')
P.legend(loc='upper left', prop={'size':'medium'}, frameon=False, ncol=2)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.ylabel(r"$P_N \propto \,T^2_\mathrm{sys} \times f(\nu) / N_b N_d$", fontsize=20.)
P.xlabel("z", fontsize=18)
P.tight_layout()
P.show()
