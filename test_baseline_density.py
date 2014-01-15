#!/usr/bin/python
"""
Rescale baseline density files, Nring(u), into n(x) = n(u=d/lambda) / lambda^2, 
which is approx. const. with frequency (and x = u / nu).
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.interpolate
import os, sys

def process_baseline_file(fname):
    """
    Process one of Prina's SKA n(u) files and output n(d), which can then be 
    converted into a freq.-dep. n(u).
    """
    # Extract info. about baseline file
    fname_end = fname.split("/")[-1]
    tmp = fname_end.split("_")
    
    freq = float(tmp[1]) / 1e6 # Freq., MHz
    dec = float(tmp[2][3:]) # Declination, degrees
    ts = float(tmp[3].split("sec")[0]) # Single baseline integ. time, in sec
    if len(tmp) == 6:
        du = float(tmp[5][2:-4]) # u bin width, ~1/sqrt(fov)
    else:
        du = float(tmp[4][2:-4]) # u bin width, ~1/sqrt(fov)
    
    # Output information about this datafile
    print "-"*50
    print "Filename:", fname
    print "-"*50
    print "Ndish:   ", Ndish
    print "Freq.:   ", freq, "MHz"
    print "Dec.:    ", dec, "deg"
    print "t_s:     ", ts, "sec"
    print "du:      ", du
    print "-"*50
    
    # Load datafile and convert to density (don't need bin centres; just use edges)
    u, Nring = np.genfromtxt(fname).T
    n = Nring / (2.*np.pi * u * du) / (24.*3600. / ts) # Eq. 18 of Mario's notes
    
    print "\n\n\n"
    print "u", u[:10]
    print "n", n[:10]
    print "N", Nring[:10]
    print "\n\n\n"
    
    # Remove leading zeros (or other special values), if any, to ensure 
    # interpolation has a sharp cut at u_min
    minidx = None; jj = -1
    while minidx is None:
        jj += 1
        if (n[jj] != 0.) and (not np.isnan(n[jj])) and (not np.isinf(n[jj])):
            minidx = jj
    print minidx
    u = u[minidx:]
    n = n[minidx:]
    
    print "\n\n\n"
    print "u", u[:10]
    print "n", n[:10]
    print "N", Nring[:10]
    print "\n\n\n"
    
    # Integrate n(u) to find normalisation (should be N_dish^2)
    norm = scipy.integrate.simps(2.*np.pi*n*u, u)
    print "Renormalising n(u) by factor of", 0.5 * Ndish * (Ndish - 1) / norm
    n *= 0.5 * Ndish * (Ndish - 1) / norm
    
    # Convert to freq.-independent expression, n(x) = n(u) * nu^2,
    # where nu is in MHz.
    n_x = n * freq**2.
    x = u / freq
    return x, n_x


infile1 = "array_config/SKAMREF2_minu/SKAMREF2_1.01e9_dec90_60sec.MS_bin34.3950_du7.75667.txt"
infile2 = "array_config/SKAMREF2_du/SKAMREF2_1.01e9_dec90_60sec.MS_du7.75667.txt"

infile3 = "array_config/SKAMREF2COMP_minu/SKAMREF2COMP_1.01e9_dec90_60sec.MS_bin60.0230_du7.75667.txt"
infile4 = "array_config/SKAMREF2COMP_du/SKAMREF2COMP_1.01e9_dec90_60sec.MS_du7.75667.txt"

dat1 = np.genfromtxt(infile1).T
dat2 = np.genfromtxt(infile2).T
dat3 = np.genfromtxt(infile3).T
dat4 = np.genfromtxt(infile4).T

"""
print "***** minu:", dat1[0,:6]
print "           ", dat1[1,:6]
print "***** du__:", dat2[0,:6]
print "           ", dat2[1,:6]
print "***** minu:", dat3[0,6:12]
print "           ", dat3[1,6:12]
print "***** du__:", dat4[0,6:12]
print "           ", dat4[1,6:12]
"""


# Process input file
Ndish = 254
x1, n_x1 = process_baseline_file(infile1)
x2, n_x2 = process_baseline_file(infile2)
x3, n_x3 = process_baseline_file(infile3)
x4, n_x4 = process_baseline_file(infile4)

# Interpolate n(u)
interp1 = scipy.interpolate.interp1d( x1, n_x1, kind='linear', 
                                      bounds_error=False, fill_value=0. )
interp2 = scipy.interpolate.interp1d( x2, n_x2, kind='linear', 
                                      bounds_error=False, fill_value=0. )
interp3 = scipy.interpolate.interp1d( x3, n_x3, kind='linear', 
                                      bounds_error=False, fill_value=0. )
interp4 = scipy.interpolate.interp1d( x4, n_x4, kind='linear', 
                                      bounds_error=False, fill_value=0. )


P.subplot(211)
ii = 30
P.plot(dat1[0][:ii], dat1[1][:ii], marker='.', label="SKAMREF2_minu")
P.plot(dat2[0][:ii], dat2[1][:ii], marker='.', label="SKAMREF2_du")
P.plot(dat3[0][:ii], dat3[1][:ii], marker='.', label="SKAMREF2COMP_minu")
P.plot(dat4[0][:ii], dat4[1][:ii], marker='.', label="SKAMREF2COMP_du", color='y')

#for i in range(ii): P.axvline(dat1[0][i], color='b', alpha=0.4)
#for i in range(ii): P.axvline(dat2[0][i], color='g', alpha=0.4)
#for i in range(ii): P.axvline(dat3[0][i], color='r', alpha=0.4)
#for i in range(ii): P.axvline(dat4[0][i], color='y', alpha=0.4)

P.xlabel(r"$u$", fontsize=20)
P.ylabel(r"$N_\mathrm{ring}$", fontsize=20)

#P.xscale('log')
#P.yscale('log')
P.legend(loc='upper right')

P.subplot(212)
P.plot(x1[:ii], n_x1[:ii], 'b.')
P.plot(x2[:ii], n_x2[:ii], 'g.')
P.plot(x3[:ii], n_x3[:ii], 'r.')
P.plot(x4[:ii], n_x4[:ii], 'y.')

xx = np.linspace(0., x4[ii], 1000)
P.plot(xx, interp1(xx), 'b-', alpha=0.5)
P.plot(xx, interp2(xx), 'g-', alpha=0.5)
P.plot(xx, interp3(xx), 'r-', alpha=0.5)
P.plot(xx, interp4(xx), 'y-', alpha=0.5)
P.xlabel(r"$x = u / \nu$", fontsize=20)
P.ylabel(r"$n^\prime(x) = n(u) \nu^2$", fontsize=20)
P.ylim((-1., 1.9e4))

#P.xscale('log')
#P.yscale('log')

P.show()

# Output to disk
#np.savetxt(outfile, np.column_stack((x, n_x)))
#print "Done."
