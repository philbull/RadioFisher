#!/usr/bin/python
"""
Rescale baseline density files, Nring(u), into n(x) = n(u=d/lambda) / lambda^2, 
which is approx. const. with frequency (and x = u / nu).
"""
import numpy as np
import pylab as P
import scipy.integrate
import os, sys

try:
    Ndish = int(sys.argv[1])
    infile = sys.argv[2]
    outfile = sys.argv[3]
except:
    print "Expects 3 arguments: Ndish, infile, outfile"
    sys.exit(1)

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
    
    # Remove leading zeros (or other special values), if any, to ensure 
    # interpolation has a sharp cut at u_min
    minidx = None; jj = -1
    while minidx is None:
        jj += 1
        if (n[jj] != 0.) and (not np.isnan(n[jj])) and (not np.isinf(n[jj])):
            minidx = jj
    u = u[minidx:]
    n = n[minidx:]
    
    # Integrate n(u) to find normalisation (should be N_dish^2)
    norm = scipy.integrate.simps(2.*np.pi*n*u, u)
    print "n(u) renorm. factor:", 0.5 * Ndish * (Ndish - 1) / norm
    n *= 0.5 * Ndish * (Ndish - 1) / norm
    
    # Convert to freq.-independent expression, n(x) = n(u) * nu^2,
    # where nu is in MHz.
    n_x = n * freq**2.
    x = u / freq
    return x, n_x


# Process input file
x, n_x = process_baseline_file(infile)

# Output to disk
np.savetxt(outfile, np.column_stack((x, n_x)))
print "Done."
