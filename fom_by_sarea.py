#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""
import numpy as np
import radiofisher as rf
import os, sys

cosmo = rf.experiments.cosmo

sarea = [50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
names = ['exptS', 'aexptM', 'exptL', 'iexptL', 'cexptL', 'GBT', 'Parkes', 
         'GMRT', 'WSRT', 'APERTIF', 'VLBA', 'JVLA', 'iJVLA', 'BINGO', 'iBAOBAB32',
         'iBAOBAB128', 'yCHIME', 'iAERA3', 'iMFAA', 'yTIANLAIpath', 'yTIANLAI', 
         'yTIANLAIband2', 'FAST', 'KAT7', 'iKAT7', 'cKAT7', 'MeerKATb1', 'iMeerKATb1',
         'cMeerKATb1', 'MeerKATb2', 'iMeerKATb2', 'cMeerKATb2', 'ASKAP', 
         'SKA1MIDbase1', 'iSKA1MIDbase1', 'cSKA1MIDbase1', 'SKA1MIDbase2', 
         'iSKA1MIDbase2', 'cSKA1MIDbase2', 'SKA1MIDfull1', 'iSKA1MIDfull1',
         'cSKA1MIDfull1', 'SKA1MIDfull2', 'iSKA1MIDfull2', 'cSKA1MIDfull2',
         'fSKA1SURbase1', 'fSKA1SURbase2', 'fSKA1SURfull1', 'fSKA1SURfull2', 
         'exptCV', 'GBTHIM', 'SKA0MID', 'fSKA0SUR', 'SKA1MID900', 'SKA1MID350',
         'iSKA1MID900', 'iSKA1MID350', 'fSKA1SUR650', 'fSKA1SUR350', 'aSKA1LOW',
         'SKAMID_PLUS', 'SKAMID_PLUS2', 'yCHIME_nocut', 'yCHIME_avglow',]

# Get expt. ID from commandline
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    print "Need to choose an experiment ID:"
    for i in range(len(names)):
        print "%2d  %s" % (i, names[i])
    sys.exit()
#print names[k]

fom = []
for s in sarea:
  try:
    root = "output/%s_scan_%d" % (names[k], s)

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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8', 'gamma']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Add Planck prior
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = rf.euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pwa = lbls.index('wa')
    
    # Calculate FOM
    fom.append( rf.figure_of_merit(pw0, pwa, None, cov=cov_pl) )
  except:
    # Fisher matrices not found or error, write None
    fom.append( np.nan )

# Output results
print "-"*50
print names[k]
print "-"*50
print "sarea  fom"
for i in range(len(sarea)):
    first = "*" if fom[i] == np.max(fom) else ""
    print ">> %15s %5d: %5.2f %s" % (names[k], sarea[i], fom[i], first)
