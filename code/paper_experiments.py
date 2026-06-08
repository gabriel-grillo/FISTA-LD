import os
# For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"
from datetime import datetime
from itertools import product
import numpy as np
import tensorflow as tf

### Tensorflow settings
# Setting random seeds for 'random', 'numpy' and 'tensorflow'
tf.keras.utils.set_random_seed(42)
# Enabling operations determinism
tf.config.experimental.enable_op_determinism()
# GPU settings
TRAIN_ON_GPU     = True
MEMORY_GROWTH    = False
# TOTAL_GPU_MEMORY = 16303
TOTAL_GPU_MEMORY = 4096
if TRAIN_ON_GPU:
    gpus = tf.config.list_physical_devices( 'GPU' )
    if gpus:
        # Setting memory growth
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth( gpu, MEMORY_GROWTH )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        # Setting memory limit
        tf.config.set_logical_device_configuration(
            gpus[0],
            [ tf.config.LogicalDeviceConfiguration( memory_limit = TOTAL_GPU_MEMORY * 0.80 ) ]
        )
else:
    tf.config.set_visible_devices( [], 'GPU' )

from traintest import test_untrained, train_fista_ld, test_fista_ld, train_deepopt, test_deepopt

#### Date to save results
# DATE = datetime.today().strftime('%Y-%m-%d')
DATE = '0'

#######################################################################################################################
#################################################### EXPERIMENT 1 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
# DATASET             = 'mayo_clinic_512'
DATASET             = 'mayo_clinic_128'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 100

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA       = np.logspace(-3.5, -1.5, 5)
GAMMA       = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
CONST_TAU   = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
PARAMS      = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
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

#### TRAINING - TESTING DEEPOPT
ALPHA_DO = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
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

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 2 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
# DATASET             = 'mayo_clinic_512'
DATASET             = 'mayo_clinic_128'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20, 40, 60, 80, 100 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 100

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# same as Experiment 1

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA     = [ 10**(-2.5) ]
GAMMA     = [ 0.05 ]
CONST_TAU = [ -1.0 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
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
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 3 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
# DATASET             = 'mayo_clinic_512'
DATASET             = 'mayo_clinic_128'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 100 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'test'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 100

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA     = [ 10**(-2.5) ]
GAMMA     = [ 0.05 ]
CONST_TAU = [ -1.0 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
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

    # Only need to test (training was performed in experiment 2)    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

### TRAINING - TESTING DEEPOPT
ALPHA_DO = [ 0.5 ]
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
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 4 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
# DATASET             = 'mayo_clinic_512'
DATASET             = 'mayo_clinic_128'
PROBLEM             = 'lstv'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 200
TEST_BATCH_SIZE     = 100

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD
ALPHA     = [ 10**(-2.5) ]
GAMMA     = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
CONST_TAU = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
PARAMS      = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
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
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 5 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
# DATASET             = 'mayo_clinic_512'
DATASET             = 'mayo_clinic_128'
PROBLEM             = 'lstv'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'test'
TEST_ITERS          = 200
TEST_BATCH_SIZE     = 100

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.75 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD
ALPHA     = [ 10**(-2.5) ]
GAMMA     = [ -1.0 ]
CONST_TAU = [ 0.75 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
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

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')
