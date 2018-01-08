#!/usr/bin/python
"""
Project a Fisher matrix for f, H etc. to alpha = f.H, used for the KSZ paper.
"""
import numpy as np
import pylab as P
import radiofisher as rf
import os

# Precompute cosmo fns.
H, r, D, f = rf.background_evolution_splines(rf.experiments.cosmo)

# Load precomputed Fisher matrix
fname = "/home/phil/oslo/bao21cm/fisher_bao_EuclidRef.dat"
F0 = np.genfromtxt(fname).T

# Get parameter names from header
ff = open(fname, 'r')
line = ff.readline()[2:-1]
ff.close()
pnames = line.split(", ")

# Load redshift bin array
zc = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95,]) # Euclid

# Trim unwanted params
EXCLUDE = ['h', 'w0', 'wa', 'oc', 'ob', 'ok', 'Mnu', 'alpha_s']
F, lbls = rf.combined_fisher_matrix([F0,], pnames, exclude=EXCLUDE)

# Do parameter conversions
def project_to_alpha(F, lbls):
    
    # Create new Fisher matrix with alpha parameter included at the end
    Nz = zc.shape[0]
    Np = F.shape[0]
    Fnew = np.zeros((Np+Nz, Np+Nz))
    Fnew[:Np,:Np] = F
    
    # Calculate derivatives of (f, H) w.r.t. alpha
    dfda = f(zc)
    daparda = np.ones(zc.shape) # alpha = f/f_fid*H/H_fid = f/f_fid*alpha_par
    
    # Loop over redshift bins
    for i in range(Nz):
        # Construct diagonal (alpha-alpha) element for this bin
        pa = Np + i
        pf = lbls.index("f%d" % i)
        pp = lbls.index("apar%d" % i)
        
        Fnew[pa, pa] = F[pf, pf] * dfda[i]**2. + F[pp, pp] * daparda[i]**2. \
                     + 2. * F[pf, pp] * dfda[i] * daparda[i]
        
        # Loop over existing parameters and construct cross-terms with alpha
        # (ignores f, apar parameters)
        for p in range(Np):
            if lbls[p][0] != 'f' and lbls[p][:4] != 'apar':
                Fnew[pa, p] = F[pf, p] * dfda[i] + F[pp, p] * daparda[i]
                Fnew[p, pa] = Fnew[pa, p]
    
    # Construct new list of parameter labels
    lnew = lbls + ['alpha%d' % i for i in range(Nz)]
    return Fnew, lnew

print lbls

# Get new matrix involving alpha
Fnew, lnew = project_to_alpha(F, lbls)

# Trim unwanted parameters
excl = ['f%d' % i for i in range(zc.size)]
excl += ['apar%d' % i for i in range(zc.size)]
Fnew, lnew = rf.combined_fisher_matrix([Fnew,], lnew, exclude=excl)

# Invert and print errors
cov = np.linalg.inv(Fnew)

for l, c in zip(lnew, np.sqrt(np.diag(cov))):
    print "%8s: %5.5f" % (l, c)

print "-"*50

cov2 = np.linalg.inv(F)
for l, c in zip(lbls, np.sqrt(np.diag(cov2))):
    print "%8s: %5.5f" % (l, c)


print "-"*50
print zc
print np.sqrt(np.diag(cov))[-len(zc):]

#print lnew
#rf.plot_corrmat(Fnew, lnew)
#P.show()
