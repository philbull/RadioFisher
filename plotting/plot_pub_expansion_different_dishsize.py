#!/usr/bin/python
"""
Plot sigma(D_A) and sigma(H) as a fn. of z, for different dish sizes
"""
import numpy as np
import pylab as P

# Load sigmas and zc
zc = np.load("expansion-dishes-zc.npy")
sig = []
sig.append(np.load("expansion-dishes-sigmas-10.0.npy"))
sig.append(np.load("expansion-dishes-sigmas-13.5.npy"))
sig.append(np.load("expansion-dishes-sigmas-15.0.npy"))
sig.append(np.load("expansion-dishes-sigmas-15.1.npy")) # actuall interferom

Hc = np.load("expansion-dishes-H.npy") # = H/100 km/s/Mpc
da_c = np.load("expansion-dishes-da.npy") # dA in Gpc

# Figure-out where the redshift bins are for D_A and H
# A, bHI, f(z), sigma_NL, aperp(z), apar(z)
zfns = [5, 4, 2]
Nbins = zc.size

col = ['r', 'b', 'y', 'g']
name = ["10m", "13.5m", "15m", "15m interf."]

# Create axes and hide labels
bigAxes = P.axes(frameon=False)
P.xticks([]); P.yticks([]) # Hide ticks
P.ylabel("Fractional error", fontdict={'fontsize':'22'})


# Upper panel
ax_top = P.subplot(211)
for i in range(len(sig)):
    sigma = sig[i]
    sigma_H = sigma[3+2*Nbins:]
    P.plot(zc, sigma_H/Hc, lw=1.5, color=col[i], label=name[i], marker='.')

P.ylabel("$\sigma/H$", fontdict={'fontsize':'20'})
##P.ylim((0.002, 0.032))
P.xlim((0.94*np.min(zc), np.max(zc)*1.005))

# Lower panel
ax_bot = P.subplot(212)
for i in range(len(sig)):
    sigma = sig[i]
    sigma_da = sigma[3+Nbins:3+2*Nbins]
    P.plot(zc, sigma_da/da_c, lw=1.5, color=col[i], label=name[i], marker='.')

P.xlabel("$z$", fontdict={'fontsize':'20'})
P.ylabel("$\sigma/D_A$", fontdict={'fontsize':'20'})
##P.ylim((0.002, 0.032)) #P.ylim((0.002, 0.024))
P.xlim((0.94*np.min(zc), np.max(zc)*1.005))

# Legend
ax_top.legend(loc='upper left', prop={'size':'x-large'})

# Move subplots
# pos = [[x0, y0], [x1, y1]]
pos1 = ax_top.get_position().get_points()
pos2 = ax_bot.get_position().get_points()
dy = pos1[0,1] - pos2[1,1]
l = pos1[0,0]
w = pos1[1,0] - pos1[0,0]
h = pos1[1,1] - pos1[0,1]
b = pos1[0,1]

ax_top.set_position([l, b - 0.5*dy, w, h+0.5*dy])
ax_bot.set_position([l, b - h - dy, w, h+0.5*dy])

# Display options
fontsize = 18.
j = 0
for tick in ax_top.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
  if j==0: tick.label1.set_visible(False)
  if j % 2 == 1: tick.label1.set_visible(False)
  j += 1

j = 0
for tick in ax_bot.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
  if j % 2 == 1: tick.label1.set_visible(False)
  j += 1

for tick in ax_top.xaxis.get_major_ticks():
  tick.label1.set_visible(False)
for tick in ax_bot.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
