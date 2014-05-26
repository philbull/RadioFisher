#!/usr/bin/python
"""
Plot effective volume as a function of perp/parallel k.
"""

import numpy as np
import pylab as P
import scipy.interpolate
import matplotlib.cm, matplotlib.ticker
import scipy.ndimage

# Load data; Veff = cs/cn / (1 + cs/cn)
Veff_sd = np.load("snr-Veff-sd.npy")
kperp_sd = np.load("snr-kperp-sd.npy")
kpar_sd = np.load("snr-kpar-sd.npy")
cn_sd = np.load("snr-cn-sd.npy")
cs_sd = np.load("snr-cs-sd.npy")

Veff_int = np.load("snr-Veff-int.npy")
kperp_int = np.load("snr-kperp-int.npy")
kpar_int = np.load("snr-kpar-int.npy")
cn_int = np.load("snr-cn-int.npy")
cs_int = np.load("snr-cs-int.npy")


locs_perp_sd = [i for i in np.arange(kperp_sd.size)[::100]]
lbls_perp_sd = ["%3.1e" % kk for kk in kperp_sd[::100]]
locs_par_sd = [i for i in np.arange(kpar_sd.size)[::100]]
lbls_par_sd = ["%3.1e" % kk for kk in kpar_sd[::100]]

locs_perp_int = [i for i in np.arange(kperp_int.size)[::100]]
lbls_perp_int = ["%3.1e" % kk for kk in kperp_int[::100]]
locs_par_int = [i for i in np.arange(kpar_int.size)[::100]]
lbls_par_int = ["%3.1e" % kk for kk in kpar_int[::100]]



Xsd, Ysd = np.meshgrid(np.log10(kperp_sd), np.log10(kpar_sd))
Xint, Yint = np.meshgrid(np.log10(kperp_int), np.log10(kpar_int))

# Smooth data
Zsd = np.log10(Veff_sd)
Zsd[np.where(np.isinf(Zsd))] = -100.
#Zsd = scipy.ndimage.filters.gaussian_filter(Zsd, sigma=10.)

Zint = np.log10(Veff_int)
Zint[np.where(np.isinf(Zint))] = -100.
#Zint2 = scipy.ndimage.filters.gaussian_filter(Zint, sigma=0.4)

print ">>>", 10.**np.max(Zsd), 10.**np.max(Zint)

P.subplot(111)
lvls = [-3., -2., -1., np.log10(0.5), np.log10(0.9), 0.001]
lws = np.ones(len(lvls)) * 1.5

# Single Dish
P.contourf(Yint.T, Xint.T, Zint.T, levels=lvls, alpha=1.0, cmap=matplotlib.cm.Blues)

# Interferom.
cnt = P.contour(Ysd.T, Xsd.T, Zsd.T, levels=lvls, linewidths=lws, alpha=1., colors='k', linestyles='solid')
#P.clabel(cnt)

# Axis labels and limits
P.xlabel("$k_\parallel \, [\mathrm{Mpc}^{-1}]$", fontsize=20.)
P.ylabel("$k_\perp \, [\mathrm{Mpc}^{-1}]$", fontsize=20.)
P.ylim((-4.05, 0.1))
P.xlim((-4.05, 0.1))


z = 1.05
l = 3e8 / (1420e6/(1.+z))
r = 3424.
k_fov = 16.*np.log(2.)/1.22 * 13.5 / (r * l)
kfg = 0.0012159813857
k_fov = 0.0238854346229
#P.axvline(np.log10(1./7), lw=3., color='k', alpha=0.3) # NL scale
#P.axvline(np.log10(kfg), lw=3., color='k', alpha=0.3)
#P.axhline(np.log10(k_fov), lw=3., color='k', alpha=0.3)

# Reformat tick labels
fmt = lambda x, y: "$10^{%d}$" % x
majorFormatter = matplotlib.ticker.FuncFormatter(fmt)
P.gca().xaxis.set_major_formatter(majorFormatter)
P.gca().yaxis.set_major_formatter(majorFormatter)


majorLocator = matplotlib.ticker.MultipleLocator(1)
P.gca().xaxis.set_major_locator(majorLocator)
P.gca().yaxis.set_major_locator(majorLocator)

P.gca().tick_params(axis='both', which='major', labelsize=22, size=8., width=1.5, pad=8.)


# Axis label font size
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()

#P.gcf().set_size_inches(8.5,10.)
P.savefig('pub-veff.pdf', transparent=True)

P.show()

"""
P.subplot(221)
P.imshow(np.log10(cn_sd), vmin=-10., vmax=10.)
P.ylabel("kpar"); P.xlabel("kperp")
P.xticks(locs_perp_sd, lbls_perp_sd)
P.yticks(locs_par_sd, lbls_par_sd)
P.colorbar()
"""
