"""
Wrapper to add the radiofisher module in parent directory to PYTHONPATH
"""
def rfimport():
    import os, sys, inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

if __name__ == 'rfwrapper':
    rfimport()
    import radiofisher as rf
