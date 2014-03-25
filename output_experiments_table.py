#!/usr/bin/python
"""
Output a LaTeX table of experimental parameters.
"""
import numpy as np
import baofisher
import experiments as e
from units import *

expts = [ e.exptS, e.exptM, e.exptL, e.GBT, e.BINGO, e.WSRT, e.APERTIF, 
          e.JVLA, e.ASKAP, e.KAT7, e.MeerKAT_band1, e.MeerKAT, e.SKA1MID,
          e.SKA1SUR, e.SKA1SUR_band1, e.SKAMID_PLUS, e.SKAMID_PLUS_band1, 
          e.SKASUR_PLUS, e.SKASUR_PLUS_band1 ]
names = ['exptS', 'iexptM', 'cexptL', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
         'JVLA', 'cASKAP', 'cKAT7', 'cMeerKAT_band1', 'cMeerKAT', 'cSKA1MID',
         'SKA1SUR', 'SKA1SUR_band1', 'SKAMID_PLUS', 'SKAMID_PLUS_band1', 
         'SKASUR_PLUS', 'SKASUR_PLUS_band1'] #, 'EuclidRef', 'EuclidOpt']
labels = ['Snapshot', 'Mature', 'Facility', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
         'JVLA', 'ASKAP', 'KAT7', 'MeerKAT (Band 1)', 'MeerKAT', 'SKA1-MID',
         'SKA1-SUR', 'SKA1-SUR (Band 1)', 'SKA1-MID+', 'SKA1-MID+ (Band 1)', 
         'SKA1-SUR+', 'SKA1-SUR+ (Band 1)'] #, 'Euclid (ref.)', 'Euclid (opt.)']


# Column headings
headings = ['$T_\mathrm{inst} \, [\mathrm{K}]$', r'$N_d \times N_b$', 
            '$D_\mathrm{dish} \, [\mathrm{m}]$', '$D_\mathrm{min} \, [\mathrm{m}]$', 
            '$D_\mathrm{max} \, [\mathrm{m}]$', r'$\nu_\mathrm{max} \, [\mathrm{MHz}]$',
            r'$\Delta\nu \, [\mathrm{MHz}]$', '$z_\mathrm{min}$', '$z_\mathrm{max}$', 
            '$S_\mathrm{area} \,[\mathrm{deg}^2]$']

# Table header
tbl = []
tbl.append("\hline")
tbl.append(r"{\bf Experiments} & " + " & ".join(headings) + " \\\\")
tbl.append("\hline")


# Add lines to table
for i in range(len(expts)):
    e = baofisher.overlapping_expts(expts[i])
    try:
        # Collect experimental parameters
        vals = []
        vals += ["%d" % (e['Tinst']*1e-3),]
        vals += [r"$%d \times %d$" % (e['Ndish'], e['Nbeam']),]
        vals += ["%3.1f" % e['Ddish'],]
        # Dmin, Dmax
        Dmin = "%3.1f" % e['Dmin'] if 'Dmin' in e.keys() else "--"
        Dmax = "%3.1f" % e['Dmax'] if 'Dmax' in e.keys() else "--"
        vals += [Dmin, Dmax]
        vals += ["%4.0f" % e['survey_numax'],]
        vals += ["%4.0f" % e['survey_dnutot'],]
        vals += ["%3.2f" % (e['nu_line']/e['survey_numax'] - 1.),]
        vals += ["%3.2f" % (e['nu_line']/(e['survey_numax'] - e['survey_dnutot']) - 1.),]
        Sarea = e['Sarea'] / (D2RAD)**2.
        vals += ["%3.0f,000" % (Sarea/1e3) ,]
    
        # Construct line in table
        line = labels[i] + " & " + " & ".join(vals) + " \\\\"
    except:
        line = labels[i] + " & " + " & ".join(["--" for j in range(len(headings))]) + " \\\\"
    tbl.append(line)

# Output table
print "-"*50
print ""
for t in tbl:
    print t
print ""
