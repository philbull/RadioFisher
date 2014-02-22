#!/usr/bin/python
"""
Take the DETF Planck Fisher matrix (technical note from arXiv:0901.0721) and project it onto more useful variables.

The DETF Planck Fisher matrix has the following parameters:
{n_s, w_m, w_b, w_k, w_DE, deltaGamma, M, logG0, log A_S, {w_i}}
where w_X = Omega_X h^2, and w_i = w(z_i). deltaGamma, M, and logG0 are 
unconstrained by the Planck matrix, so must be trimmed. w_m is the *total* 
matter density (CDM + baryons).

We want to project to the following parameters:
{n_s, Omega_M, [Omega_b], Omega_K, Omega_DE, h, w0, wa, log A_S}
(we don't care about Omega_b, but we should keep it in so we can marginalise over it)

The DETF Planck Fisher matrix can be obtained from:
http://c3.lbl.gov:8000/Trac.Cosmology/browser/Programs/FoMSWG/tags/original/DATA/PLANCK.dat?rev=842
"""
import numpy as np
import pylab as P

# Load DETF Planck Fisher matrix (indexes start from 0)
dat = np.genfromtxt("DETF_PLANCK_FISHER.txt").T
N = np.max(dat[0]) + 1
F = np.zeros((N,N))
for k in range(dat.shape[1]):
    i = dat[0,k]
    j = dat[1,k]
    F[i,j] = dat[2,k]


# Define fiducial values (from p8 of arXiv:0901.0721)
n_s = 0.963
w_m = 0.1326
w_b = 0.0227
w_k = 0.
w_DE = 0.3844
h = 0.719
fid = np.zeros(N)
fid[:5] = [n_s, w_m, w_b, w_k, w_DE] # Ignore the rest, not needed

# Construct projection operator to new parameters, d(p_old) / d(p_new)
Nnew = 9
M = np.zeros((Nnew, N))
idx_ns, idx_h, idx_w0, idx_wa = 0, 5, 6, 7
idx_AS_old, idx_AS_new = 8, 8
idx_omega = [1, 2, 3, 4]
idx_wi = np.arange(9, N).astype(int)

# n_s and log A_S are kept the same
M[idx_ns, idx_ns] = 1.
M[idx_AS_new, idx_AS_old] = 1.

# Derivatives w.r.t. density parameters (= h^2 for all) and h (=)
for i in idx_omega:
    M[idx_h, i] = 2. * fid[i] / h # Deriv. for h
    M[i,i] = h**2. # Deriv. for Omega_X (cross-terms are zero)

# Derivatives w.r.t. EOS parameters
a0 = 0.1
da = 0.025
for i in range(len(idx_wi)):
    aa = 1. - (float(i) + 0.5)*da # Centroid of 'a' bin (p8 of arXiv:0901.0721)
    M[idx_w0, idx_wi[i]] = 1.
    M[idx_wa, idx_wi[i]] = 1. - aa

# Multiply old Fisher matrix by the projection operator
Fnew = np.dot(M, np.dot(F, M.T))
print "Condition no.:", np.linalg.cond(Fnew)

# Save to file
hdr = "0:n_s, 1:O_M, 2:O_b, 3:O_K, 4:O_DE, 5:h, 6:w0, 7:wa, 8:logA_S"
np.savetxt("fisher_detf_planck.dat", Fnew, header=hdr)
print "Saved to file."


# FIXME
import euclid
Feuc = euclid.planck_prior_full # w0, wa, omega_DE, omega_k, w_m, w_b, n_s
print "F(ns,ns): %4.4e  %4.4e" % (Fnew[0,0], Feuc[6,6])
print "F(ok,ok): %4.4e  %4.4e" % (Fnew[3,3], Feuc[3,3])
print "F(ol,ol): %4.4e  %4.4e" % (Fnew[4,4], Feuc[2,2])
print "F(w0,w0): %4.4e  %4.4e" % (Fnew[6,6], Feuc[0,0])
print "F(wa,wa): %4.4e  %4.4e" % (Fnew[7,7], Feuc[1,1])
print "F(w0,wa): %4.4e  %4.4e" % (Fnew[6,7], Feuc[0,1])
print ""
print "F(wm,wm): %4.4e  %4.4e" % (F[1,1], Feuc[4,4])
print "F(wb,wb): %4.4e  %4.4e" % (F[2,2], Feuc[5,5])
print "F(wm,wb): %4.4e  %4.4e" % (F[1,2], Feuc[4,5])
exit()
#{n_s, w_m, w_b, w_k, w_DE, deltaGamma, M, logG0, log A_S, {w_i}}


# Construct correlation matrix
F = Fnew
F_corr = np.zeros(F.shape)
for ii in range(F.shape[0]):
    for jj in range(F.shape[0]):
        F_corr[ii,jj] = F[ii, jj] / np.sqrt(F[ii,ii] * F[jj,jj])

# Plot corrmat
fig = P.figure()
ax = fig.add_subplot(111)
import matplotlib.cm
#F_corr = F_corr**3.
matshow = ax.matshow(F_corr, vmin=-1., vmax=1., cmap=matplotlib.cm.get_cmap("RdBu"))
#ax.title("z = %3.3f" % zc[i])
fig.colorbar(matshow)
P.title(hdr)
P.show()
