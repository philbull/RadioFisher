#!/usr/bin/python
"""
Check separated window function from astro-ph/9602009 Eq. 5
"""
import numpy as np
import pylab as P

# Top-hat window function
t = np.linspace(0., 1000., 10000)

# Plot results
ax1 = P.subplot(211)
ax2 = P.subplot(212)

y = []
for i in [0, 5000]: # Offsets
    # Construct window fn.
    w = np.zeros(t.shape)
    w[:50] = 1.
    w[i:i+50] = 1.
    
    # FFT tophat
    W = np.fft.ifftshift( np.fft.fft(w) )
    
    ax1.plot(w)
    #ax2.plot(W)
    y.append(W)

ax2.plot(y[1] / y[0])
ax1.set_ylim((0., 1.1))
ax2.set_xlim((4990, 5050))
P.show()
