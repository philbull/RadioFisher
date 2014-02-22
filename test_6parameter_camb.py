#!/usr/bin/python
"""
Test what happens when Tau, h are changed for CAMB.
(Needed for 6 parameter LCDM model)
"""

import numpy as np
import camb_wrapper as camb
import pylab as P

"""
As = np.logspace(-10., -8., 15) # Scalar amplitudes
sig8 = []
for i in range(As.size):
    p = {'output_root': 'TESTX', 'scalar_amp__1___':As[i]}
    camb.camb_params("TESTX.ini", **p)
    vals = camb.run_camb("TESTX.ini", camb_exec_dir="/home/phil/oslo/bao21cm/camb")
    #k1, pk1 = np.genfromtxt("camb/TESTX_matterpower.dat").T
    try:
        sig8.append(vals['sigma8'])
    except:
        print vals

np.save("sig8", np.column_stack((As, np.array(sig8))))
"""

As, sig8 = np.load("sig8.npy").T

P.plot(As, sig8**2., 'r-', marker='.')

x = [As[0], As[-1]]
y = [sig8[0]**2., sig8[-1]**2.]

P.plot(x, y, 'b-')

P.show()



exit()
# Output dictionaries
print "-"*50
print vals1
print "-"*50
print vals2
print "-"*50



P.subplot(111)
P.plot(k1, pk1, 'b-', lw=1.5)
P.plot(k2, pk2, 'g-', lw=1.5)
P.xscale('log')
P.yscale('log')
P.show()
