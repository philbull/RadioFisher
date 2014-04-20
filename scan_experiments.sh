#!/bin/bash

for i in `seq 0 42`
do
    for sarea in 100 500 1000 2000 5000 10000 15000 20000 25000
    do
        mpirun -n 1 ./full_experiment.py $i $sarea
    done
done
