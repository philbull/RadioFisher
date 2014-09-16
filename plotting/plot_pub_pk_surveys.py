#!/usr/bin/python
"""
Plot parameter 1D marginal constraints as a fn. of FG subtraction efficiency.
"""
import numpy as np
import pylab as P
import baofisher

nsurveys = 2 #5
nbins = 14

name = ['SD', 'Interferom.']
#name = ['A', 'B', 'C', 'D', 'E']
cols = ['b', 'g', 'c', 'r', 'y']

P.subplot(111)
for j in range(nsurveys):
    for i in range(1): #range(nbins): # FIXME: Just z=0.1
        #if j == 0: continue
        fname_smooth = "tmp/pk_surveys_smooth_powspec_"+str(j)+"_"+str(i)+".npy"
        fname_constr = "tmp/pk_surveys_constraints_"+str(j)+"_"+str(i)+".npy"
        
        k, pk, fbao, pksmooth = np.load(fname_smooth).T
        kc, pkc, pkerr, fbao_c, pkcsmooth = np.load(fname_constr).T
        
        pkerr[np.where(np.isinf(pkerr))] = 1e9
        #pkerr[np.where(np.isnan(pkerr))] = 1e9
        P.plot(kc, pkerr, color=cols[j], label=name[j], marker='.', lw=1.5)
        
        print pkerr
        
        # FIXME: h^-1 units?
        #P.plot(k, pk)
        #yup, ylow = baofisher.fix_log_plot(pkc, pkerr*pkc)
        #P.errorbar(kc, pkc, yerr=[ylow, yup], marker='.', ls='none', color='r')
        

P.xscale('log')
P.yscale('log')

P.xlim((4e-3, 2.))
#P.ylim((3e-3, 1e1))

# Display options
P.legend(loc='upper left', prop={'size':'x-large'}, ncol=2)
P.ylabel("$\Delta P(k) / P(k) \; (z=0)$", fontdict={'fontsize':'22'})
P.xlabel("$k \,[Mpc^{-1}]$", fontdict={'fontsize':'20'})

fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
