import os
from datetime import datetime
import numpy as np
from itertools import product

from paths import plot_path
import plots_and_metrics as pm

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_128'
TEST_DATASET_RATIO  = 0.05
PROBLEM             = 'lasso'

# TRANING PARAMETERS
DATE                = datetime.today().strftime('%Y-%m-%d')
EPOCHS_TO_SAVE      = [ 5 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 0.05 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]

# FISTA-LD PARAMETERS
ALPHA         = [10**(-2.5)]
GAMMA         = [ 0.05 ]
CONST_TAU     = [ -1.0 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = [ 0.5 ]
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### DATA PATH ##################################
DATA_PATH = plot_path( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### SAVING RESULTS
results_matrix = np.stack( [ d.get('data') for d in mean_rel_opt_gap_dicts ] ).T
np.savetxt( os.path.join( DATA_PATH, 'optgap.txt' ), results_matrix )
with open( os.path.join( DATA_PATH, 'optgap_labels.txt' ), 'w' ) as f:
    for d in mean_rel_opt_gap_dicts:
        f.write(f"{d.get('label')}\n")
#### PLOTTING
pm.plot_data( mean_rel_opt_gap_dicts,
              ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
              fig_path = DATA_PATH, fig_name = 'optgap' )

######################### PLOTTING MEAN (F-F*)/F* -- only trained iterations ##################################
#### COPYING (F-F*)/F* DICTS AND UPDATING DATA
zoom_mean_rel_opt_gap_dicts = [ d.copy() for d in mean_rel_opt_gap_dicts ]
for i,d in enumerate(zoom_mean_rel_opt_gap_dicts):
    d.update( { 'data':mean_rel_opt_gap_dicts[i].get('data')[:max(MAXITER)] } )
#### PLOTTING
pm.plot_data( zoom_mean_rel_opt_gap_dicts,
              ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
              fig_path = DATA_PATH, fig_name = 'optgap_zoom' )

######################### PLOTTING MEAN R ##################################
#### LOADING R DATA
R_dicts = pm.load_data( 'Rhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, [] )
#### COMPUTING MEAN R
R_mean_dicts = pm.compute_mean( R_dicts )
#### STACKING O(1/k**2)
iters = R_mean_dicts[0].get('data').shape[0]
R_mean_dicts = [ dict( data = 1e+6/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + R_mean_dicts
#### STACKING O(1/k**3)
R_mean_dicts = [ dict( data = 1e+9/(1 + np.arange(iters))**3, label = '$O(1/k^3)$', ls = '--', c = 'k' ) ] + R_mean_dicts
#### STACKING THE X-COORDINATE
R_mean_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + R_mean_dicts
#### SAVING RESULTS
results_matrix = np.stack( [ d.get('data') for d in R_mean_dicts ] ).T
np.savetxt( os.path.join( DATA_PATH, 'R.txt' ), results_matrix )
with open( os.path.join( DATA_PATH, 'R_labels.txt' ), 'w' ) as f:
    for d in R_mean_dicts:
        f.write(f"{d.get('label')}\n")
#### PLOTTING
pm.plot_data( R_mean_dicts,
              ylabel = r'$\mean_b\left(R_{b,k}\right)$',
              fig_path = DATA_PATH, fig_name = 'R' )

######################### PLOTTING DEVIATION NORMS ##################################
#### LOADING |dy| DATA -- FISTA-LD
dy_norm_dicts = pm.compute_mean( pm.load_data( 'dyhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], FISTA_LD_LIST, [] ) )
#### LOADING |dw| DATA -- FISTA-LD
dw_norm_dicts = pm.compute_mean( pm.load_data( 'dwhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], FISTA_LD_LIST, [] ) )
#### LOADING |dx1| DATA -- DEEPOPT
dx1_norm_dicts = pm.compute_mean( pm.load_data( 'dx1hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], [], DEEPOPT_LIST ) )
#### LOADING |dx2| DATA -- DEEPOPT
dx2_norm_dicts = pm.compute_mean( pm.load_data( 'dx2hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], [], DEEPOPT_LIST ) )
#### STACKING ALL DEVIATIONS
d_norms = dy_norm_dicts + dw_norm_dicts + dx1_norm_dicts + dx2_norm_dicts
#### STACKING O(1/k)
iters = d_norms[0].get('data').shape[0]
d_norms = [ dict( data = 1e+3/(1 + np.arange(iters)), label = '$O(1/k)$', ls = ':', c = 'k' ) ] + d_norms
#### STACKING O(1/k**1.5)
d_norms = [ dict( data = 3e+2/(1 + np.arange(iters)), label = r'$O(1/k^{1.5})$', ls = '-.', c = 'k' ) ] + d_norms
#### STACKING THE X-COORDINATE
d_norms = [ dict( data = 1 + np.arange(iters), label = '$k$', ls = None, c = None ) ] + d_norms
#### PLOTTING
pm.plot_data( d_norms,
              ylabel = r'$\mean_b\left(\|\text{deviation}_{b,k}\|_2\right)$',
              fig_path = DATA_PATH, fig_name = 'deviations_norm' )

######################### PLOTTING UPPER BOUNDS TO DEVIATION NORMS ##################################
## TO DO

######################### PLOTTING DEVIATION NORMS REL TO UPPER BOUNDS ##################################
## TO DO

######################### TABLE 1 ##################################
## TO DO

######################### TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

######################### LSTV ANALYSIS ##################################
## TO DO


print( 'All data was saved to:', DATA_PATH )