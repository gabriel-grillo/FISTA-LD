import os
# For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"
import time
from datetime import datetime
from itertools import product
import tensorflow as tf

### Tensorflow settings
# Setting random seeds for 'random', 'numpy' and 'tensorflow'
tf.keras.utils.set_random_seed(42)
# Enabling operations determinism
tf.config.experimental.enable_op_determinism()
# GPU settings
TRAIN_ON_GPU = True
if not TRAIN_ON_GPU:
    tf.config.set_visible_devices( [], 'GPU' )

from traintest import test_untrained, train_fista_ld, test_fista_ld, train_deepopt, test_deepopt

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

##### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lasso'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso'
DATE                = datetime.today().strftime('%Y-%m-%d')

TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 5 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val' # 'test' or 'val'
TEST_ITERS          = 100
TEST_BATCH_SIZE     = 11

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [0.05] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]

# FISTA-LD PARAMETERS -- set a negative const_tau to enable varying tau
ALPHA     = [ 10**(-2.5) ]
GAMMA     = [ 0.05 ]
CONST_TAU = [ -1.0 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DO = [ 0.5 ]

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

#### TEST UNTRAINED ALGORITHMS
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD 
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, alpha, gamma, const_tau,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

##### TRAINING - TESTING DEEPOPT
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), ALPHA_DO ) ):
    dataset_ratio, iters, alpha = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    miniter, maxiter = iters

    label = 'DeepOpt ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_deepopt( DATASET, PROBLEM, miniter, maxiter, alpha,
                   EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                   dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_deepopt( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, epochs, DATE,
                      TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
