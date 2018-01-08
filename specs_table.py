#!/usr/bin/python
"""
Output table of instrumental specifications.
"""
import numpy as np
import radiofisher as rf
e = rf.experiments

expt_list = [
    ( 'exptS',            e.exptS ),        # 0
    ( 'iexptM',           e.exptM ),        # 1
    ( 'exptL',            e.exptL ),        # 2
    ( 'iexptL',           e.exptL ),        # 3
    ( 'cexptL',           e.exptL ),        # 4
    ( 'GBT',              e.GBT ),          # 5
    ( 'Parkes',           e.Parkes ),       # 6
    ( 'GMRT',             e.GMRT ),         # 7
    ( 'WSRT',             e.WSRT ),         # 8
    ( 'APERTIF',          e.APERTIF ),      # 9
    ( 'VLBA',             e.VLBA ),         # 10
    ( 'JVLA',             e.JVLA ),         # 11
    ( 'iJVLA',            e.JVLA ),         # 12
    ( 'BINGO',            e.BINGO ),        # 13
    ( 'iBAOBAB32',        e.BAOBAB32 ),     # 14
    ( 'iBAOBAB128',       e.BAOBAB128 ),    # 15
    ( 'yCHIME',           e.CHIME ),        # 16
    ( 'iAERA3',           e.AERA3 ),        # 17
    ( 'iMFAA',            e.MFAA ),         # 18
    ( 'yTIANLAIpath',     e.TIANLAIpath ),  # 19
    ( 'yTIANLAI',         e.TIANLAI ),      # 20
    ( 'yTIANLAIband2',    e.TIANLAIband2 ), # 21
    ( 'FAST',             e.FAST ),         # 22
    ( 'KAT7',             e.KAT7 ),         # 23
    ( 'iKAT7',            e.KAT7 ),         # 24
    ( 'cKAT7',            e.KAT7 ),         # 25
    ( 'MeerKATb1',        e.MeerKATb1 ),    # 26
    ( 'iMeerKATb1',       e.MeerKATb1 ),    # 27
    ( 'cMeerKATb1',       e.MeerKATb1 ),    # 28
    ( 'MeerKATb2',        e.MeerKATb2 ),    # 29
    ( 'iMeerKATb2',       e.MeerKATb2 ),    # 30
    ( 'cMeerKATb2',       e.MeerKATb2 ),    # 31
    ( 'ASKAP',            e.ASKAP ),        # 32
    ( 'SKA1MIDbase1',     e.SKA1MIDbase1 ), # 33
    ( 'iSKA1MIDbase1',    e.SKA1MIDbase1 ), # 34
    ( 'cSKA1MIDbase1',    e.SKA1MIDbase1 ), # 35
    ( 'SKA1MIDbase2',     e.SKA1MIDbase2 ), # 36
    ( 'iSKA1MIDbase2',    e.SKA1MIDbase2 ), # 37
    ( 'cSKA1MIDbase2',    e.SKA1MIDbase2 ), # 38
    ( 'SKA1MIDfull1',     e.SKA1MIDfull1 ), # 39
    ( 'iSKA1MIDfull1',    e.SKA1MIDfull1 ), # 40
    ( 'cSKA1MIDfull1',    e.SKA1MIDfull1 ), # 41
    ( 'SKA1MIDfull2',     e.SKA1MIDfull2 ), # 42
    ( 'iSKA1MIDfull2',    e.SKA1MIDfull2 ), # 43
    ( 'cSKA1MIDfull2',    e.SKA1MIDfull2 ), # 44
    ( 'fSKA1SURbase1',    e.SKA1SURbase1 ), # 45
    ( 'fSKA1SURbase2',    e.SKA1SURbase2 ), # 46
    ( 'fSKA1SURfull1',    e.SKA1SURfull1 ), # 47
    ( 'fSKA1SURfull2',    e.SKA1SURfull2 ), # 48
    ( 'exptCV',           e.exptCV ),       # 49
    ( 'GBTHIM',           e.GBTHIM ),       # 50
    ( 'SKA0MID',          e.SKA0MID ),      # 51
    ( 'fSKA0SUR',         e.SKA0SUR ),      # 52
    ( 'SKA1MID900',       e.SKA1MID900 ),   # 53
    ( 'SKA1MID350',       e.SKA1MID350 ),   # 54
    ( 'iSKA1MID900',      e.SKA1MID900 ),   # 55
    ( 'iSKA1MID350',      e.SKA1MID350 ),   # 56
    ( 'fSKA1SUR650',      e.SKA1SUR650 ),   # 57
    ( 'fSKA1SUR350',      e.SKA1SUR350 ),   # 58
    ( 'aSKA1LOW',         e.SKA1LOW ),      # 59
    ( 'SKAMID_PLUS',      e.SKAMID_PLUS ),  # 60
    ( 'SKAMID_PLUS2',     e.SKAMID_PLUS2 )  # 61
]

for (name, expt) in expt_list:
    try:
        Ndish = expt['Ndish']
        Nbeam = expt['Nbeam']
        Ddish = expt['Ddish']
        Tinst = expt['Tinst'] / 1e3
        dnutot = expt['survey_dnutot']
        numax = expt['survey_numax']
        numin = numax - dnutot
        
        zmax = 1420. / numin - 1.
        zmin = 1420. / numax - 1.
        
        line = r" & & %s & %d & $%d \times %d$ & %3.1f & -- & -- & %d & %d & %d & %3.2f & % 3.2f & x \\" \
           % (name, Tinst, Ndish, Nbeam, Ddish, numax, numin, dnutot, zmin, zmax)
    except:
        line = r"%s \\" % name    
    print line
    
