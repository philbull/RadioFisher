#!/usr/bin/python
import numpy as np
import configparser
import json

def trim_comments(val):
    """
    Remove in-line comments that begin with "#", and whitespace.
    """
    return val.split("#")[0].strip()

def parse(val):
    """
    Process the returned value to trim comments and identify special commands 
    and types.
    
    Possible special commands are:
    
      * linspace(): Maps to numpy.linspace
      * logspace:   Maps to numpy.logspace
    
    Only the basic options of the corresponding numpy commands are supported.
    """
    # Trim trailing comments
    val = trim_comments(val)
    
    # Check for booleans and NoneType
    if val.lower() == 'true': return True
    if val.lower() == 'false': return False
    if val.lower() == 'none': return None
    
    # Check for numbers (all numbers interpreted as floating point)
    try:
        out = float(val)
        return out
    except:
        out = val
    
    # Process lists
    if "[" in val:
        val = json.loads(val)
        return val
    
    # Look for special commands
    if "linspace(" in val:
        xmin, xmax, xnum = val[9:-1].split(",")
        return np.linspace(float(xmin), float(xmax), int(xnum))
    
    if "logspace(" in val:
        xmin, xmax, xnum = val[9:-1].split(",")
        return np.logspace(float(xmin), float(xmax), int(xnum))
    
    return out

def load_config(fname, debug=False):
    """
    Load configuration from a specified file and return processed results in a 
    dictionary. (Set debug=True to see debugging output from parser.)
    """
    # Initialise ConfigParser and start processing
    cfg = configparser.ConfigParser()
    cfg.read(fname)
    
    # Loop through sections and settings
    settings = {}
    for sect in cfg.sections():
        for opt in cfg.options(sect):
            val = parse( cfg.get(sect, opt) )
            settings[opt] = val
            if debug: print("%15s: |%s| %s" % (opt, val, type(val)))
    return settings

