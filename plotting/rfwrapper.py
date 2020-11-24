"""
Wrapper to add the radiofisher module in parent directory to PYTHONPATH
"""
def rfimport():
    import os, sys, inspect
    
    # Add parent directory to PYTHONPATH
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    
    # Change directory if 
    if "/plotting" in os.getcwd():
        print("rfwrapper: Changed working directory to %s" % parentdir)
        os.chdir(parentdir)

if __name__ == 'rfwrapper':
    rfimport()
    import radiofisher as rf
