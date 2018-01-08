#!/usr/bin/python
"""
Test derivative of D(a) with respect to f(a).
"""
import numpy as np
import pylab as P
import radiofisher as rf
import copy

# Load cosmo functions
H, r, D, f = rf.background_evolution_splines(rf.experiments.cosmo)
a = np.linspace(1./(1.+6.), 1., 1000)
z = 1./a - 1.

# Fitting fns. to f, D
#ff = np.poly1d( np.polyfit(a, f(z), deg=14) )
#DD = np.poly1d( np.polyfit(a, D(z), deg=14) )

#df_da = ff.deriv(1)(a)
df_da = np.gradient(f(z), a[1]-a[0])

#dD_da = DD.deriv(1)
dD_da = np.gradient(D(z), a[1]-a[0])

# FIXME: Check to see which gives correct answer for dF_RSD/dgamma!
def Frsd(a, gamma=0.55, mu=1.):
    # RSD function to be used for numerical derivatives
    c = copy.deepcopy(rf.experiments.cosmo)
    c['gamma'] = gamma
    H, r, D, f = rf.background_evolution_splines(c)
    z = 1./a - 1.
    
    # F_RSD
    b = 1. #np.sqrt(1.+z)
    Frsd = (b + mu**2.*f(z))**2. * D(z)**2.
    return Frsd

def DD(a, gamma=0.55):
    # RSD function to be used for numerical derivatives
    c = copy.deepcopy(rf.experiments.cosmo)
    c['gamma'] = gamma
    H, r, D, f = rf.background_evolution_splines(c)
    z = 1./a - 1.
    return D(z)

def ff(a, gamma=0.55):
    # RSD function to be used for numerical derivatives
    c = copy.deepcopy(rf.experiments.cosmo)
    c['gamma'] = gamma
    H, r, D, f = rf.background_evolution_splines(c)
    z = 1./a - 1.
    return f(z)

def Frsd_deriv(a, gamma=0.55, mu=1.):
    # Analytic deriv. used in baofisher code
    H, r, D, f = rf.background_evolution_splines(rf.experiments.cosmo)
    z = 1./a - 1.
    b = 1. #np.sqrt(1.+z)
    
    #ff = np.poly1d( np.polyfit(a, f(z), deg=6) )
    #df_da = ff.deriv(1)(a)
    df_da = np.gradient(f(z), a[1]-a[0])
    
    #dD_da = D(z)*f(z)/a
    dD_da = np.gradient(D(z), a[1]-a[0])
    
    # Get derivative of F_RSD w.r.t. f(z)
    dD_df = dD_da / df_da
    #dD_df = np.log(a)
    dF_df = 2. * (b + f(z)*mu**2.) * mu**2. * D(z)**2. \
          + 2. * (b + f(z)*mu**2.)**2. * D(z) * dD_df
    
    # Chain rule, for deriv. w.r.t. gamma
    logOma = np.log(f(z)) / gamma
    return dF_df, dF_df * f(z) * logOma


##########
#for col, g in zip(['r', 'y', 'b'], [0.5, 0.55, 0.6]):
#    c = copy.deepcopy(rf.experiments.cosmo)
#    c['gamma'] = g
#    H, r, D, f = rf.background_evolution_splines(c)
#    P.plot(z, D(z), lw=2., color=col)
#    P.plot(z, f(z), lw=2., color=col, ls='dashed')

#P.show()
#exit()
##########


# Numerical derivative of F_RSD
gam = 0.55
dgam = 0.001
dFdg = (Frsd(a, gam+dgam) - Frsd(a, gam-dgam)) / (2.*dgam)

# dF/df
da = 0.001
dF_df = (Frsd(a+da) - Frsd(a-da)) / (2.*da) / df_da

# Analytic deriv. in code
dFdf_ana, dFdg_ana = Frsd_deriv(a)

# Plotting
P.subplot(111)

# dF/dgamma
#P.plot(z, dFdg, 'k-', lw=2., label="dF/d$\gamma$ Num.")
#P.plot(z, dFdg_ana, 'r--', lw=2., label="dF/d$\gamma$ Ana.")

# dF_RSD / df
#P.plot(z, dF_df, 'b-', lw=2., label="dF/df Num.")
#P.plot(z, dFdf_ana, 'y--', lw=2., label="dF/df Ana.")

df_da = np.gradient(f(z), a[1]-a[0])
dD_da = D(z)*f(z) / a

# dD/dgamma
dfdg_ana = f(z) * np.log(f(z)) / gam
dDdg_ana = (dD_da / df_da) * dfdg_ana
dDdg = (DD(a, gam+dgam) - DD(a, gam-dgam)) / (2.*dgam) # numerical deriv.
P.plot(z, dDdg_ana, 'm-', lw=2., label="dD/d$\gamma$ Ana.")
P.plot(z, dDdg, 'c--', lw=2., label="dD/d$\gamma$ Num.")

# df/dgamma
dfdg = (ff(a, gam+dgam) - ff(a, gam-dgam)) / (2.*dgam) # numerical deriv.
#P.plot(z, dfdg_ana, 'g-', lw=2., label="df/d$\gamma$ Ana.")
#P.plot(z, dfdg, 'y--', lw=2., label="df/d$\gamma$ Num.")

P.legend(loc='upper right', ncol=2, frameon=False)
#P.ylim((-2., 2.))
P.show()
exit()



# Deriv: dD/df ~ (dD/da) / (df/da)
P.plot(a, dD_da(a) / df_da(a), 'k-')

# Found by inverting f = dlogD/dloga:
P.plot(a, -DD(a) / ff(a), 'r-')

#P.plot(a, D(z)*f(z) / a / df_da(a), 'k--')
#P.plot(a, ddisc, 'c-')
P.plot(a, np.log(a), 'g--')
P.plot(a, DD(a)*ff(a)/a / df_da(a), 'y--')

P.show()
