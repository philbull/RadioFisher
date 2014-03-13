#!/usr/bin/python
"""
Triangle plot in matplotlib
"""
import numpy as np
import pylab as P

Nparam = 6 # No. of parameters in triangle plot

# Add axes to figure
fig = P.figure()
axes = [[fig.add_subplot(Nparam, Nparam, (j+1) + i*Nparam) for i in range(j, Nparam)] for j in range(Nparam)]

# Fixed width and height for each subplot
w = 1.0 / (Nparam+1.)
h = 1.0 / (Nparam+1.)
l0 = 0.1
b0 = 0.1

x = np.linspace(-1., 1., 100)
y = x**2.

lbls = ['a', 'b', 'c', 'd', 'e', 'f']

# Loop through rows, columns, repositioning plots
# i is column, j is row
for j in range(Nparam):
    for i in range(Nparam-j):
        ax = axes[j][i]
        pos = ax.get_position().get_points()
        ax.set_position([l0+w*i, b0+h*j, w, h])
        
        if j != 0:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_visible(False)
        if i != 0:
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_visible(False)
        
        if i == 0: ax.plot(x, y)
        
        #if j == 0: ax.set_xlabel(lbls[i])
        if i == Nparam-j-1: ax.set_title(lbls[i])
        if i == 0: ax.set_ylabel(lbls[Nparam-j-1])
P.show()
