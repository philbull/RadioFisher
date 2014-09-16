#!/usr/bin/python
"""
Plot sigma(fNL) as a fn. of z, for different fiducial fNL
"""
import numpy as np
import pylab as P
import experiments
import baofisher

# Load sigma(fNL) and z bins
fNL = [0., 1., 10.]
sigmas = []; sigmas_bz = []
sigma_fNL = []; sigma_fNL_bz = []
sigma_bias = []

for j in range(len(fNL)):
    zc = np.load("Xnongauss-fnl-zc-%d.npy"%j)
    sigmas.append(np.load("tmp/Xnongauss-fnl-%d.npy" % j))
    sigmas_bz.append(np.load("tmp/nongauss-fnl-zbias-%d.npy" % j))
    sigma_fNL.append(sigmas[j][-zc.size:])
    sigma_fNL_bz.append(sigmas_bz[j][-zc.size:])
    sigma_bias.append(sigmas_bz[j][1:zc.size+1])

col = ['r', 'b', 'y']
name = ["f_NL = " + str(int(_fNL)) for _fNL in fNL]

# Get bias as fn. of z
cosmo = experiments.cosmo
cosmo['bHI0'] = 0.702
bias = baofisher.bias_HI(zc, cosmo)

P.subplot(111)
for i in range(len(fNL)):
    #P.errorbar(zc, fNL[i]*np.ones(zc.size), yerr=sigma_fNL[i], marker='.', color=col[i], lw=1.5, label=name[i])
    P.plot(zc, sigma_fNL[i], lw=1.5, color=col[i], marker='.', label=name[i])
    #P.plot(zc, sigma_fNL_bz[i], lw=1.5, color=col[i], marker='.', ls='dashed')
#P.plot(zc, bias - 1., 'k-')
#P.axhline(0., ls='dotted', color='k')

#P.xlim((np.min(zs), np.max(zs)))
P.ylim((0., 150.))
P.xlim((0., 3.6))

# Display options
P.legend(loc='upper right', prop={'size':'x-large'})
P.ylabel("$\sigma_{fNL}(z)$", fontdict={'fontsize':'22'})
P.xlabel("$z$", fontdict={'fontsize':'20'})

fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)


"""
P.subplot(212)
for i in range(len(fNL)):
    P.plot(zc, sigma_bias[i])
"""
P.tight_layout()
P.show()
