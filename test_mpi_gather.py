#!/usr/bin/python
"""
Test MPI gathering vectors.
"""
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()


F = [0 for i in range(2*size)]
F[myid] = (myid + 1.) * np.eye(2)
Tnew = comm.gather(F, [], root=0)

if myid == 0:
    Ftot = np.sum(Tnew, axis=0)
    print Ftot
    print np.shape(Ftot)
    #print Tnew
    #print np.sum(Tnew, axis=1)

