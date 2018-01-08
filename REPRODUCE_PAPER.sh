#!/bin/bash
# 
# Run all code necessary to reproduce figures and results in Bull et al. (2014)
# (arXiv:1405.1452). This script will not necessarily run all the way through 
# first time -- comments explain where manual intervention is necessary. Also, 
# running all of the forecasts can take a substantial amoount of time.
# 
# Phil Bull (philbull@gmail.com)
# Updated: 2014-12-21

# Number of CPUs to use for MPI
NCPU=15

# Change to correct working directory
cd bao21cm/

################################################################################
# Run forecasts for reference models
################################################################################

# Run forecasts for IM reference models
mpirun -n $NCPU python2.7 ./full_experiment.py 0
mpirun -n $NCPU python2.7 ./full_experiment.py 1
mpirun -n $NCPU python2.7 ./full_experiment.py 2

# Run forecast for DETF IV reference model
mpirun -n $NCPU python2.7 ./galaxy_full_experiment.py 1

# Run forecasts for Facility ref. model with different distance measures 
# turned on and off
mpirun -n $NCPU python2.7 ./scan_distance_measures.py 2

# Scan through all relevant survey/experiment params for Stage II and Facility
# (for Figs. 21-26) [takes a long time]
./run_all_scan_experiment.sh 1
./run_all_scan_experiment.sh 2


################################################################################
# Make plots
################################################################################

# Fig. 1: fig01-scales.pdf
# (This figure is just an illustration)

# Fig. 2: fig02-veff.pdf
# N.B. Uncomment the code block in radiofisher/baofisher.py:fisher_integrands()
# to generate the data to make this plot. Run full_survey.py for SKA1MID350 
# (id=54) and iSKA1MID350 (id=56), changing the line in fisher_integrands() 
# between 'mode = "sd"' and 'mode = "int"' depending on whether single-dish or 
# interferometer mode. Then run the plotting script.
python2.7 ./plotting/plot_Veff.py

# Fig. 3: fig03-resolution-z.pdf
python2.7 ./plotting/plot_resolution_redshift.py

# Fig. 4: fig04-dlogp.pdf
python2.7 ./plotting/plot_dlogp.py

# Fig. 5: fig05-fbao.pdf
python2.7 ./plotting/plot_fbao.py

# Fig. 6: fig06-zfns.pdf
python2.7 ./plotting/plot_all_redshift_functions.py

# Fig. 7: fig07-dlogp-fnz.pdf
python2.7 ./plotting/plot_dlogp_fnz.py

# Fig. 8: fig08-zfns-distance-measures.pdf
python2.7 ./plotting/plot_redshift_functions_different_distance_measures.py

# Fig. 9: fig09-lss-distances.pdf
python2.7 ./plotting/plot_lss_distances.py

# Fig. 10: fig10-fs8-bs8.pdf
python2.7 ./plotting/plot_fs8_bs8.py

# Fig. 11: fig11-5params.pdf
python2.7 ./plotting/plot_triangle_6params.py

# Fig. 12: fig12-w0wa-with-without-ok.pdf
python2.7 ./plotting/plot_w0wa_with_without_curvature.py

# Fig. 13: fig13-w0omegaDE-okfixed.pdf
python2.7 ./plotting/plot_omegaDEw0.py

# Fig. 14: fig14-fom-improvement.pdf
python2.7 ./plotting/plot_omegak_improvement.py

# Fig. 15: fig15-ok-improvement.pdf
python2.7 ./plotting/plot_omegak_improvement.py

# Fig. 16: fig16-ok.pdf
python2.7 ./plotting/plot_1d_omegak.py

# Fig. 17: fig17-6params-eos.pdf
python2.7 ./plotting/plot_triangle_5eosparams.py

# Fig. 18: fig18-gamma-improvement.pdf
python2.7 ./plotting/plot_omegak_improvement.py

# Fig. 19: fig19-w0gamma.pdf
python2.7 ./plotting/plot_w0gamma.py

# Fig. 20: fig20-omegaHI-evol.pdf
python2.7 ./plotting/plot_omegaHI_data.py

# Fig. 21: fig21-omegaHI.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 3

# Fig. 22: fig22-signl.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 5

# Fig. 23: fig23-efg.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 2

# Fig. 24: fig24-kfg.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 4

# Fig. 25: fig25-ttot.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 0

# Fig. 26: fig26-sarea.pdf
python2.7 ./plotting/plot_expt_parameter_scan_multipanel.py 1

# Fig. 27: fig27-dlogp-ideal.pdf
python2.7 ./plotting/plot_ideal_dlogp.py

# Fig. 28: fig28-w0wa-combined.pdf
python2.7 ./plotting/plot_w0wa_combined.py

# Fig. 29: fig29-pk-lowk.pdf
python2.7 ./plotting/plot_pk.py

# Fig. 30: fig30-beams.pdf
python2.7 ./plotting/plot_beams.py

