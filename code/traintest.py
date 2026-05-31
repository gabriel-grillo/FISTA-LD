import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as pp
pp.rcParams[ 'savefig.bbox' ] = 'tight'

from operators import reconstruction_operators, torch_linear_op_as_tf2_layer, wavelet_as_tf2_layer
from data import get_dataset
from optimization import fista, ista, fista_ld, computeR, deepopt_nonsmooth

from itertools import product

#######################################################################################################################
#################################################### GPU SETTINGS #####################################################
#######################################################################################################################
TRAIN_ON_GPU = True
MEMORY_GROWTH = False
if TRAIN_ON_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Setting memory growth
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, MEMORY_GROWTH)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        # # Setting memory limit
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=16303 * 0.90)]
        # )
else:
    tf.config.set_visible_devices([], 'GPU')


#######################################################################################################################
################################################ PROBLEM'S DEFINITION #################################################
#######################################################################################################################
def problem_definition( dataset, mode, problem, batch_size, dataset_ratio ):
    ##### OPERATORS FOR RECONSTRUCTION (OPTIMIZATION)
    ray_trafo, ray_trafo_transp, img_shape, sino_shape, \
        ray_trafo_norm, reg_trafo, reg_trafo_transp, reg_trafo_norm, lam = reconstruction_operators( dataset, problem )
    
    ##### DOMAIN SHAPE
    x_shape = img_shape + (1,) # one channel
    
    ##### TENSORFLOW INTEGRATION
    A = torch_linear_op_as_tf2_layer( ray_trafo, ray_trafo_transp, img_shape, sino_shape, np.float32, name = 'ray_transform' )
    A_adj = torch_linear_op_as_tf2_layer( ray_trafo_transp, ray_trafo, sino_shape, img_shape, np.float32, name = 'ray_transform_adj' )
    A_norm = tf.constant( ray_trafo_norm, dtype = tf.float32, name = 'A_norm' )
    if problem == 'lasso':
        W = wavelet_as_tf2_layer( reg_trafo, reg_trafo_transp, img_shape, np.float32, name = 'regularization_transform' )
        W_adj = wavelet_as_tf2_layer( reg_trafo_transp, reg_trafo, img_shape, np.float32, name = 'regularization_transform_adj' )
        lam = tf.constant( lam[None,...,None], dtype = tf.float32, name = 'lam' )
    else:
        W = reg_trafo
        W_adj = reg_trafo_transp
        lam = tf.constant( lam, dtype = tf.float32, name = 'lam' )
    W_norm = tf.constant( reg_trafo_norm, dtype = tf.float32, name = 'W_norm' )
    
    
    ###### REFERENCE FUNCTIONAL VALUES FOR TRAINING
    if problem == 'lstv':
        alpha_FISTA   = tf.constant( 10**(-2.5), dtype = tf.float32 )
        R0_FISTA      = tf.constant(       1e+7, dtype = tf.float32 )
        tau_FISTA     = tf.constant(        0.5, dtype = tf.float32 )
        maxiter_FISTA = tf.constant(        200, dtype = tf.int32   )
    else:
        alpha_FISTA   = tf.constant(  1.0, dtype = tf.float32 )
        tau_FISTA     = tf.constant(  1.0, dtype = tf.float32 )
        R0_FISTA      = tf.constant(  0.0, dtype = tf.float32 )
        maxiter_FISTA = tf.constant( 1000, dtype = tf.int32   )
    if dataset == 'mayo_clinic_512':
        num_detectors = 1000
        num_angles = 1000
    elif dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
    else:
        raise ValueError()
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),))
    def F_ref( b ):
        # Initial point
        x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Iterates FISTA saving functional values
        _, _, hist = fista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam, maxiter_FISTA,
                            alpha_FISTA, tau_FISTA, R0_FISTA,
                            save_hist = True
                            )
        # Returns only the functional values history
        return hist[0]
    
    ##### DATASET
    ds = get_dataset( dataset, mode, problem, batch_size, F_ref, overall_ratio = dataset_ratio )

    return ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape

#######################################################################################################################
################################################### TRAIN ROUTINES ####################################################
#######################################################################################################################

def training_loop( epochs, train_step, ds_train, save_weights, saving_path, load_saved_info = False, epochs_to_save = [] ):
    ##### LOADING TRAINING INFO
    if load_saved_info:
        with open( os.path.join( saving_path, 'log.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            initial_epoch = int(lines[0])
        with open( os.path.join( saving_path, 'hist.dat' ), 'r' ) as file:
            hist = np.loadtxt( file )
            hist = np.concatenate( (hist, np.zeros((2,epochs))), axis = 1 )
            train_loss_hist = hist[0]
            time_hist = hist[1]
    else:
        initial_epoch = 0
        train_loss_hist = np.zeros(epochs)
        time_hist = np.zeros(epochs)

    ##### TRAINING LOOP
    for epoch in range(initial_epoch,initial_epoch+epochs):
        start_time = time.time()

        # Iterate over the batches of the dataset.
        mean_train_loss = 0.0
        for ( b_train, F_b ) in ds_train:
            # Training step
            loss = train_step( b_train, F_b )
            # Mean train loss accumulation
            mean_train_loss += loss.numpy()
        mean_train_loss /= float( ds_train.cardinality().numpy() )
        train_loss_hist[epoch] = mean_train_loss

        time_hist[epoch] = time.time() - start_time

        # Printing new info
        print( '--------------------------------------------------------------------------------' )
        print( "Results of epoch %d" % (epoch,) )
        print( "Mean train loss at epoch %d: %.4f" % ( epoch, mean_train_loss ) )
        print( "Time taken: %.2fs" % ( time_hist[epoch] ) )

        # Saving the weights every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) in epochs_to_save:
            save_weights(epoch + 1)
    # Saving final weights
    save_weights(epoch + 1)

    if epochs > 0:
        ##### PRINTING FINAL TRAINING INFO
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        print( f'Total number of epochs: {epoch+1}' )
        print( f'Average training time per epoch (s): { np.sum(time_hist[1:]) / (epochs-1) }' )
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        ##### PLOTTING HIST INFO
        pp.semilogy(train_loss_hist[:epoch+1], marker = 'o', label = 'Train loss')
        pp.grid()
        pp.legend()
        pp.savefig( os.path.join( saving_path, 'train_hist.png' ) )
        pp.close()
        ##### SAVING FINAL TRAINING INFO
        if os.path.exists( os.path.join( saving_path, 'hist.dat' ) ):
            os.remove( os.path.join( saving_path, 'hist.dat' ) )
        with open( os.path.join( saving_path, 'hist.dat' ), 'ab' ) as file:
            hist = np.vstack( (train_loss_hist, time_hist) )
            np.savetxt( file, hist[:,:epoch+1] )
        if os.path.exists( os.path.join( saving_path, 'log.dat' ) ):
            os.remove( os.path.join( saving_path, 'log.dat' ) )
        with open( os.path.join( saving_path, 'log.dat' ), 'ab' ) as file:
            np.savetxt( file, [epoch+1] )
            np.savetxt( file, [np.sum(time_hist[1:]) / (epochs-1)] )


def train_fista_ld( dataset, problem, miniter, maxiter, alpha, gamma, const_tau,
                    epochs, batch_size, date, load_saved_models, 
                    dataset_ratio = 1.0, epochs_to_save = [] ):
    ######################### PROBLEM'S DEFINITION ##################################
    ds_train, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
        problem_definition( dataset, 'train', problem, batch_size, dataset_ratio )

    ######################### MODELS' PATH ##################################
    MODEL_PATH = os.path.join( os.getcwd(), 'params', 'fista-ld', dataset, str( dataset_ratio * 100 ) + 'percent', problem )

    if maxiter == miniter:
        trained_iterations = 'fixed'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif maxiter > miniter:
        trained_iterations = 'random'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()

    if const_tau <= 0:
        var_tau = True
        MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:]  )
    else:
        var_tau = False
        MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:], 'const_tau' + str(const_tau)[2:]  )
    MODEL_PATH = os.path.join( MODEL_PATH, 'date_' + date )

    if os.path.exists(MODEL_PATH) and not load_saved_models:
        for ( dirpath, dirnames, filenames ) in os.walk(MODEL_PATH):
            for fi in filenames:
                os.remove( os.path.join( dirpath, fi ) )

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    ######################### MODELS' DEFINITION ##################################
    ##### DY MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'log.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dy_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dy.keras' ) )
    else:
        input_y = keras.Input(shape=x_shape, name='y')
        input_grad_prev = keras.Input(shape=x_shape, name='grad_prev')
        input_dy_prev = keras.Input(shape=x_shape, name='dy_prev')

        input_convnet = keras.layers.Concatenate(axis=-1)([input_y, input_grad_prev, input_dy_prev])
        
        FILTERS = 32
        KSIZE = 3
        N_OUT = 1
        N_LAYERS = 2

        x = keras.layers.GroupNormalization(groups=-1)(input_convnet)
        for i in range(N_LAYERS):
            x = keras.layers.Conv2D(FILTERS, KSIZE,
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.L2(1e-5),
                                    kernel_initializer=keras.initializers.VarianceScaling())(x)
            x = keras.layers.GroupNormalization(groups=-1)(x)
            x = keras.layers.LeakyReLU()(x)
        result = keras.layers.Conv2D(N_OUT, KSIZE, padding='same')(x)
        
        dy_model = keras.Model(inputs=[input_y, input_grad_prev, input_dy_prev], outputs=result, name="dy_model")
    dy_model.summary()
    ##### DW MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'log.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dw_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dw.keras' ) )
    else:
        input_y = keras.Input(shape=x_shape, name='y')
        input_grad = keras.Input(shape=x_shape, name='grad')
        input_dy = keras.Input(shape=x_shape, name='dy')
        input_dw_prev = keras.Input(shape=x_shape, name='dw_prev')
        
        input_convnet = keras.layers.Concatenate(axis=-1)([input_y, input_grad, input_dy, input_dw_prev])
        
        FILTERS = 32
        KSIZE = 3
        N_OUT = 1
        N_LAYERS = 2

        x = keras.layers.GroupNormalization(groups=-1)(input_convnet)
        for i in range(N_LAYERS):
            x = keras.layers.Conv2D(FILTERS, KSIZE,
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.L2(1e-5),
                                    kernel_initializer=keras.initializers.VarianceScaling())(x)
            x = keras.layers.GroupNormalization(groups=-1)(x)
            x = keras.layers.LeakyReLU()(x)
        result = keras.layers.Conv2D(N_OUT, KSIZE, padding='same')(x)
        
        dw_model = keras.Model(inputs=[input_y, input_grad, input_dy, input_dw_prev], outputs=result, name="dw_model")
    dw_model.summary()
    ##### SAVING INITIALIZED MODELS
    def save_weights(epoch):
        if not os.path.exists( os.path.join( MODEL_PATH, f'epoch_{epoch}') ):
            os.makedirs( os.path.join( MODEL_PATH, f'epoch_{epoch}') )
        dy_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dy.keras' ) )
        dw_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dw.keras' ) )
    # Always saves the initialized model
    if not load_saved_models:
        save_weights(0)

    ######################### TRAINING ##################################
    ##### CHOOSE OPTIMIZER
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    ##### COMPUTE R0
    if problem == 'lstv':
        R0 = 1e+7
        np.save( os.path.join( MODEL_PATH, 'R0.npy' ), R0 )
    else:
        R0 = computeR( problem, ds_train, x_shape, A, A_adj, A_norm, W, W_adj, W_norm, lam )
        np.save( os.path.join( MODEL_PATH, 'R0.npy' ), R0.numpy() )
    ##### TRAINING STEP
    if dataset == 'mayo_clinic_512':
        num_detectors = 1000
        num_angles = 1000
    elif dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
    else:
        raise ValueError()
    if problem == 'lstv':
        ref_iters = 201
    else:
        ref_iters = 1001
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, ref_iters], dtype=tf.float32)))
    def train_step( b_train, F_ref ):
        # Initial point
        x0_train  = tf.zeros( shape = (tf.shape(b_train)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        if trained_iterations == 'fixed':
            n_iter_train = tf.constant( maxiter, dtype = tf.int32, name = 'maxiter' )
        else:
            n_iter_train = tf.random.uniform( shape = [],  minval = miniter, maxval = maxiter + 1, \
                                            dtype = tf.int32, name = 'maxiter' )
        # Open a GradientTape: enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            _, F, _ = fista_ld( problem, x0_train, A, A_adj, A_norm,
                                b_train, W, W_adj, W_norm, lam, dy_model, dw_model,
                                n_iter_train,
                                tf.constant( alpha, dtype=tf.float32 ),
                                R0,
                                var_tau = var_tau,
                                gamma = tf.constant( gamma, dtype=tf.float32 ),
                                const_tau = tf.constant( const_tau, dtype=tf.float32 ),
                                training = True,
                                save_hist = False
                            )
            loss = tf.reduce_mean( ( F - F_ref[:,-1] ) / ( F_ref[:,n_iter_train] - F_ref[:,-1] ) )
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, dy_model.trainable_weights + dw_model.trainable_weights)
        # Run one step of optimizer
        optimizer.apply_gradients( zip( grads, dy_model.trainable_weights + dw_model.trainable_weights ) )
        return loss
    ##### TRAINING LOOP
    training_loop( epochs, train_step, ds_train, save_weights, MODEL_PATH, load_saved_info = load_saved_models, epochs_to_save = epochs_to_save )


def train_deepopt( dataset, problem, miniter, maxiter, alpha,
                   epochs, batch_size, date, load_saved_models,
                   dataset_ratio = 1.0, epochs_to_save = [] ):
    ######################### PROBLEM'S DEFINITION ##################################
    ds_train, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
        problem_definition( dataset, 'train', problem, batch_size, dataset_ratio )
    
    ######################### MODELS' PATH ##################################
    MODEL_PATH = os.path.join( os.getcwd(), 'params', 'deepopt', dataset, str( dataset_ratio * 100 ) + 'percent', problem )
    if maxiter == miniter:
        trained_iterations = 'fixed'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif maxiter > miniter:
        trained_iterations = 'random'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()

    MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:] )
    MODEL_PATH = os.path.join( MODEL_PATH, 'date_' + date )

    if os.path.exists(MODEL_PATH) and not load_saved_models:
        for ( dirpath, dirnames, filenames ) in os.walk(MODEL_PATH):
            for fi in filenames:
                os.remove( os.path.join( dirpath, fi ) )

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    ######################### MODELS' DEFINITION ##################################
    ##### DX1 MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'log.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dx1_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dx1.keras' ) )
    else:
        input_x = keras.Input(shape=x_shape, name='x')
        input_grad = keras.Input(shape=x_shape, name='grad')
        input_dx1 = keras.Input(shape=x_shape, name='dx1')
        
        input_convnet = keras.layers.Concatenate(axis=-1)([input_x, input_grad, input_dx1])
        
        FILTERS = 32
        KSIZE = 3
        N_OUT = 1
        N_LAYERS = 2

        x = keras.layers.GroupNormalization(groups=-1)(input_convnet)
        for i in range(N_LAYERS):
            x = keras.layers.Conv2D(FILTERS, KSIZE,
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.L2(1e-5),
                                    kernel_initializer=keras.initializers.VarianceScaling())(x)
            x = keras.layers.GroupNormalization(groups=-1)(x)
            x = keras.layers.LeakyReLU()(x)
        result = keras.layers.Conv2D(N_OUT, KSIZE, padding='same')(x)
        
        dx1_model = keras.Model(inputs=[input_x, input_grad, input_dx1], outputs=result, name="dx1_model")
    dx1_model.summary()

    ##### DX2 MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'log.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dx2_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dx2.keras' ) )
    else:
        input_x = keras.Input(shape=x_shape, name='x')
        input_grad = keras.Input(shape=x_shape, name='grad')
        input_dx2 = keras.Input(shape=x_shape, name='dx2')
        input_dx1 = keras.Input(shape=x_shape, name='dx1')
        
        input_convnet = keras.layers.Concatenate(axis=-1)([input_x, input_grad, input_dx2, input_dx1])
        
        FILTERS = 32
        KSIZE = 3
        N_OUT = 1
        N_LAYERS = 2
        
        x = keras.layers.GroupNormalization(groups=-1)(input_convnet)
        for i in range(N_LAYERS):
            x = keras.layers.Conv2D(FILTERS, KSIZE,
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.L2(1e-5),
                                    kernel_initializer=keras.initializers.VarianceScaling())(x)
            x = keras.layers.GroupNormalization(groups=-1)(x)
            x = keras.layers.LeakyReLU()(x)
        result = keras.layers.Conv2D(N_OUT, KSIZE, padding='same')(x)
        
        dx2_model = keras.Model(inputs=[input_x, input_grad, input_dx2, input_dx1], outputs=result, name="dx2_model")
    dx2_model.summary()
    ##### SAVING INITIALIZED MODELS
    def save_weights(epoch):
        if not os.path.exists( os.path.join( MODEL_PATH, f'epoch_{epoch}') ):
            os.makedirs( os.path.join( MODEL_PATH, f'epoch_{epoch}') )
        dx1_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dx1.keras' ) )
        dx2_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dx2.keras' ) )
        # Always saves the initialized model
    if not load_saved_models:
        save_weights(0)

    ######################### TRAINING ##################################
    ##### CHOOSE OPTIMIZER
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    ##### TRAINING STEP
    if dataset == 'mayo_clinic_512':
        num_detectors = 1000
        num_angles = 1000
    elif dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
    else:
        raise ValueError()
    if problem == 'lstv':
        ref_iters = 201
    else:
        ref_iters = 1001
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, ref_iters], dtype=tf.float32)))
    def train_step( b_train, F_ref ):
        # Initial point
        x0_train  = tf.zeros( shape = (tf.shape(b_train)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        if trained_iterations == 'fixed':
            n_iter_train = tf.constant( maxiter, dtype = tf.int32, name = 'maxiter' )
        else:
            n_iter_train = tf.random.uniform( shape = [],  minval = miniter, maxval = maxiter + 1, \
                                                dtype = tf.int32, name = 'maxiter')
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            _, F, _ = deepopt_nonsmooth( problem, x0_train, A, A_adj, A_norm, b_train, W, W_adj, W_norm, lam, \
                                            dx1_model, dx2_model, n_iter_train, tf.constant(alpha, dtype=tf.float32), \
                                            training = True, save_hist = False)
            loss = tf.reduce_mean( ( F - F_ref[:,-1] ) / ( F_ref[:,n_iter_train] - F_ref[:,-1] ) )
        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, dx1_model.trainable_weights + dx2_model.trainable_weights)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip( grads, dx1_model.trainable_weights + dx2_model.trainable_weights ) )
        return loss

    ##### TRAINING LOOP
    training_loop( epochs, train_step, ds_train, save_weights, MODEL_PATH, load_saved_info = load_saved_models, epochs_to_save = epochs_to_save )


#######################################################################################################################
################################################### TEST ROUTINES #####################################################
#######################################################################################################################
def test_untrained( dataset, problem, mode, algorithm, batch_size, iters, tau = 1.0, dataset_ratio = 1.0 ):
    assert algorithm == 'fista' or algorithm == 'ista', 'Unimplemented algorithm: ' + algorithm
    ######################### PROBLEM'S DEFINITION ##################################
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, dataset_ratio )

    ######################### RESULTS' PATH ##################################
    if algorithm == 'fista':
        RESULT_PATH = os.path.join( os.getcwd(), 'test_results', mode, 'untrained', algorithm, \
                                    'tau' + str(tau)[2:], dataset, str( dataset_ratio * 100 ) + 'percent', problem )
    else:
        RESULT_PATH = os.path.join( os.getcwd(), 'test_results', mode, 'untrained', algorithm, \
                                    dataset, str( dataset_ratio * 100 ) + 'percent', problem )
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs( RESULT_PATH )
        os.makedirs( os.path.join( RESULT_PATH, 'image' ) )

    Fhist_filename = os.path.join( RESULT_PATH, 'Fhist.npy' )
    if os.path.exists(Fhist_filename):
        os.remove(Fhist_filename)
    Rhist_filename = os.path.join( RESULT_PATH, 'Rhist.npy' )
    if os.path.exists(Rhist_filename):
        os.remove(Rhist_filename)
    flag_hist_filename = os.path.join( RESULT_PATH, 'flag_hist.npy' )
    if os.path.exists(flag_hist_filename):
        os.remove(flag_hist_filename)
    inner_iter_hist_filename = os.path.join( RESULT_PATH, 'inner_iter_hist.npy' )
    if os.path.exists(inner_iter_hist_filename):
        os.remove(inner_iter_hist_filename)

    ######################### TEST STEP ##################################
    if algorithm == 'fista':
        if problem == 'lstv':
            alpha_FISTA = tf.constant( 10**(-2.5), dtype = tf.float32 )
            R0_FISTA    = tf.constant(       1e+7, dtype = tf.float32 )
        else:
            alpha_FISTA = tf.constant(  1.0, dtype = tf.float32 )
            R0_FISTA    = tf.constant(  0.0, dtype = tf.float32 )
        if dataset == 'mayo_clinic_512':
            num_detectors = 1000
            num_angles = 1000
        elif dataset == 'mayo_clinic_128':
            num_detectors = 183
            num_angles = 285
        else:
            raise ValueError()
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),))
        def test_step( b ):
            # Initial point
            x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
            # Number of iterations
            n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
            x, _, hist = fista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam, n_iter,
                                alpha_FISTA, tau, R0_FISTA,
                                save_hist = True
                               )
            return x, hist
    else:
        if dataset == 'mayo_clinic_512':
            num_detectors = 1000
            num_angles = 1000
        elif dataset == 'mayo_clinic_128':
            num_detectors = 183
            num_angles = 285
        else:
            raise ValueError()
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),))
        def test_step( b ):
            # Initial point
            x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
            # Number of iterations
            n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
            x, _, hist = ista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                               n_iter, save_hist = True
                             )
            return x, hist
            
    ######################### RUNING OVER DATASET ##################################
    Fhist_list = []
    Rhist_list = []
    flag_hist_list = []
    inner_iter_hist_list = []
    for step, (b,_) in enumerate(ds):
        # Running the algorithm
        rec, hist = test_step( b )

        for j in range(b.shape[0]):
            # Printing results
            print("Test example ", step * batch_size + j, "------------------------")
            print("Final loss ", hist[0][j][-1].numpy())

            # Saving reconstructions
            pp.imshow( rec[j,...,0], cmap = 'gray' )
            pp.savefig( os.path.join( RESULT_PATH, 'image', 'rec' + str(step * batch_size + j) + '.png' ) )
            pp.close()

        # Saving test info to list
        Fhist_list.append( hist[0].numpy() )
        Rhist_list.append( hist[1].numpy() )
        flag_hist_list.append( hist[2].numpy() )
        inner_iter_hist_list.append( hist[3].numpy() )

    # Saving hist info to file
    with open( Fhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( Fhist_list, axis = 0 ) )
    with open( Rhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( Rhist_list, axis = 0 ) )
    with open( flag_hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( flag_hist_list, axis = 0 ) )
    with open( inner_iter_hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( inner_iter_hist_list, axis = 0 ) )


def test_fista_ld( dataset, problem, mode, miniter, maxiter, alpha, gamma, const_tau,
                   epochs, batch_size, iters, date, dataset_ratio = (1.0,1.0) ):
    ######################### PROBLEM'S DEFINITION ##################################
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, test_dataset_ratio )

    ######################### LOADING MODELS ##################################
    MODEL_PATH = os.path.join( os.getcwd(), 'params', 'fista-ld', dataset, str( train_dataset_ratio * 100 ) + 'percent', problem )
    if maxiter == miniter:
        trained_iterations = 'fixed'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif maxiter > miniter:
        trained_iterations = 'random'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    if const_tau <= 0:
        var_tau = True
        MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:]  )
    else:
        var_tau = False
        MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:], 'const_tau' + str(const_tau)[2:]  )
    MODEL_PATH = os.path.join( MODEL_PATH, 'date_' + date )

    ######################### RESULTS' PATH ##################################
    RESULT_PATH = os.path.join( os.getcwd(), 'test_results', mode, 'trained', 'fista-ld', dataset, \
                                f'{100*train_dataset_ratio}percent_train', \
                                f'{100*test_dataset_ratio}percent_test', \
                                problem )
    if trained_iterations == 'fixed':
        RESULT_PATH = os.path.join( RESULT_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif trained_iterations == 'random':
        RESULT_PATH = os.path.join( RESULT_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    if const_tau <= 0:
        var_tau = True
        RESULT_PATH = os.path.join( RESULT_PATH, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:] )
    else:
        var_tau = False
        RESULT_PATH = os.path.join( RESULT_PATH, 'alpha' + str(alpha)[2:], 'const_tau' + str(const_tau)[2:] )
    
    RESULT_PATH = os.path.join( RESULT_PATH, 'date_' + date )
    RESULT_PATH = os.path.join( RESULT_PATH, f'epoch_{epochs}' )
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    Fhist_filename = os.path.join( RESULT_PATH, 'Fhist.npy' )
    if os.path.exists(Fhist_filename):
        os.remove(Fhist_filename)

    Rhist_filename = os.path.join( RESULT_PATH, 'Rhist.npy' )
    if os.path.exists(Rhist_filename):
        os.remove(Rhist_filename)

    dyhist_filename = os.path.join( RESULT_PATH, 'dyhist.npy' )
    if os.path.exists(dyhist_filename):
        os.remove(dyhist_filename)

    dyubhist_filename = os.path.join( RESULT_PATH, 'dyubhist.npy' )
    if os.path.exists(dyubhist_filename):
        os.remove(dyubhist_filename)

    dwhist_filename = os.path.join( RESULT_PATH, 'dwhist.npy' )
    if os.path.exists(dwhist_filename):
        os.remove(dwhist_filename)

    dwubhist_filename = os.path.join( RESULT_PATH, 'dwubhist.npy' )
    if os.path.exists(dwubhist_filename):
        os.remove(dwubhist_filename)

    flag_hist_filename = os.path.join( RESULT_PATH, 'flag_hist.npy' )
    if os.path.exists(flag_hist_filename):
        os.remove(flag_hist_filename)

    inner_iter_hist_filename = os.path.join( RESULT_PATH, 'inner_iter_hist.npy' )
    if os.path.exists(inner_iter_hist_filename):
        os.remove(inner_iter_hist_filename)

    ######################### TEST STEP ##################################
    ##### LOADING R0
    R0 = np.load( os.path.join( MODEL_PATH, 'R0.npy' ) )
    ##### LOADING THE MODELS
    MODEL_PATH = os.path.join( MODEL_PATH, f'epoch_{epochs}' )
    if os.path.exists(MODEL_PATH):
        dy_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dy.keras') )
        dw_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dw.keras') )
    else:
        raise ValueError('This algorithm has not been trained')
    if dataset == 'mayo_clinic_512':
        num_detectors = 1000
        num_angles = 1000
    elif dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
    else:
        raise ValueError()
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),))
    def test_step( b ):
        # Initial point
        x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
        _, _, hist = fista_ld( problem, x0, A, A_adj, A_norm,
                               b, W, W_adj, W_norm, lam,
                               dy_model, dw_model,
                               n_iter,
                               tf.constant( alpha, dtype=tf.float32 ),
                               tf.constant( R0, dtype=tf.float32 ),
                               var_tau = var_tau,
                               gamma = tf.constant( gamma, dtype=tf.float32 ),
                               const_tau = tf.constant( const_tau, dtype=tf.float32 ),
                               training = False,
                               save_hist = True
                              )
        return hist
    
    ######################### RUNING OVER DATASET ##################################
    Fhist_list = []
    Rhist_list = []
    dyhist_list = []
    dyubhist_list = []
    dwhist_list = []
    dwubhist_list = []
    flag_hist_list = []
    inner_iter_hist_list = []
    for step, (b,_) in enumerate(ds):
        # Running the algorithm
        hist = test_step( b )
        
        # Printing results
        for j in range(b.shape[0]):
            print("Test example ", step * batch_size + j, "------------------------")
            print("Final loss ", hist[0][j][-1].numpy())

        # Saving test info to list
        Fhist_list.append( hist[0].numpy() )
        Rhist_list.append( hist[1].numpy() )
        dyhist_list.append( hist[2].numpy() )
        dyubhist_list.append( hist[3].numpy() )
        dwhist_list.append( hist[4].numpy() )
        dwubhist_list.append( hist[5].numpy() )
        flag_hist_list.append( hist[6].numpy() )
        inner_iter_hist_list.append( hist[7].numpy() )
    
    # Saving hist info to file
    with open( Fhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( Fhist_list, axis = 0 ) )
    with open( Rhist_filename, 'wb' ) as f: 
        np.save( f, np.concatenate( Rhist_list, axis = 0 ) )
    with open( dyhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dyhist_list, axis = 0 ) )
    with open( dyubhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dyubhist_list, axis = 0 ) )
    with open( dwhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dwhist_list, axis = 0 ) )
    with open( dwubhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dwubhist_list, axis = 0 ) )
    with open( flag_hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( flag_hist_list, axis = 0 ) )
    with open( inner_iter_hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( inner_iter_hist_list, axis = 0 ) )


def test_deepopt( dataset, problem, mode, miniter, maxiter, alpha, epochs, date,
                  batch_size, iters, dataset_ratio = (1.0,1.0) ):
    ######################### PROBLEM'S DEFINITION ##################################
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, test_dataset_ratio )

    ######################### LOADING MODELS ##################################
    MODEL_PATH = os.path.join( os.getcwd(), 'params', 'deepopt', dataset, f'{100*train_dataset_ratio}percent', problem )
    if maxiter == miniter:
        trained_iterations = 'fixed'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif maxiter > miniter:
        trained_iterations = 'random'
        MODEL_PATH = os.path.join( MODEL_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    MODEL_PATH = os.path.join( MODEL_PATH, 'alpha' + str(alpha)[2:] )
    MODEL_PATH = os.path.join( MODEL_PATH, 'date_' + date )
    MODEL_PATH = os.path.join( MODEL_PATH, f'epoch_{epochs}' )

    if os.path.exists(MODEL_PATH):
        dx1_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dx1.keras') )
        dx2_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dx2.keras') )
    else:
        raise ValueError('This algorithm has not been trained')        

    ######################### RESULTS' PATH ##################################
    RESULT_PATH = os.path.join( os.getcwd(), 'test_results', mode, 'trained', 'deepopt', dataset,\
                                f'{100*train_dataset_ratio}percent_train', \
                                f'{100*test_dataset_ratio}percent_test', \
                                problem )
    if trained_iterations == 'fixed':
        RESULT_PATH = os.path.join( RESULT_PATH, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif trained_iterations == 'random':
        RESULT_PATH = os.path.join( RESULT_PATH, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    RESULT_PATH = os.path.join( RESULT_PATH, 'alpha' + str(alpha)[2:] )
    RESULT_PATH = os.path.join( RESULT_PATH, 'date_' + date )
    RESULT_PATH = os.path.join( RESULT_PATH, f'epoch_{epochs}' )
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    Fhist_filename = os.path.join( RESULT_PATH, 'Fhist.npy' )
    if os.path.exists(Fhist_filename):
        os.remove(Fhist_filename)

    dx1hist_filename = os.path.join( RESULT_PATH, 'dx1hist.npy' )
    if os.path.exists(dx1hist_filename):
        os.remove(dx1hist_filename)

    dx1ubhist_filename = os.path.join( RESULT_PATH, 'dx1ubhist.npy' )
    if os.path.exists(dx1ubhist_filename):
        os.remove(dx1ubhist_filename)

    dx2hist_filename = os.path.join( RESULT_PATH, 'dx2hist.npy' )
    if os.path.exists(dx2hist_filename):
        os.remove(dx2hist_filename)

    dx2ubhist_filename = os.path.join( RESULT_PATH, 'dx2ubhist.npy' )
    if os.path.exists(dx2ubhist_filename):
        os.remove(dx2ubhist_filename)

    ######################### TEST STEP ##################################
    ##### TEST STEP
    if dataset == 'mayo_clinic_512':
        num_detectors = 1000
        num_angles = 1000
    elif dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
    else:
        raise ValueError()
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, num_detectors, num_angles, 1], dtype=tf.float32),))
    def test_step( b ):
        # Initial point
        x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
        _, _, hist = deepopt_nonsmooth( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                                        dx1_model, dx2_model, n_iter, tf.constant(alpha, dtype=tf.float32),
                                        training = False, save_hist = True )
        return hist
    
    ######################### RUNING OVER DATASET ##################################
    Fhist_list = []
    dx1hist_list = []
    dx1ubhist_list = []
    dx2hist_list = []
    dx2ubhist_list = []
    for step, (b,_) in enumerate(ds):
        # Running the algorithm
        hist = test_step( b )
        
        # Printing results
        for j in range(b.shape[0]):
            print("Test example ", step * batch_size + j, "------------------------")
            print("Final loss ", hist[0][j][-1].numpy())

        # Saving test info to list
        Fhist_list.append( hist[0].numpy() )

        dx1hist_list.append( hist[1].numpy() )
        dx1ubhist_list.append( hist[2].numpy() )
        dx2hist_list.append( hist[3].numpy() )
        dx2ubhist_list.append( hist[4].numpy() )
    
    # Saving hist info to file
    with open( Fhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( Fhist_list, axis = 0 ) )
    with open( dx1hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dx1hist_list, axis = 0 ) )
    with open( dx1ubhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dx1ubhist_list, axis = 0 ) )
    with open( dx2hist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dx2hist_list, axis = 0 ) )
    with open( dx2ubhist_filename, 'wb' ) as f:
        np.save( f, np.concatenate( dx2ubhist_list, axis = 0 ) )


#######################################################################################################################
########################################## RUNNING TRAIN AND TEST ROUTINES ############################################
#######################################################################################################################

##### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_128'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lasso'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
DATE                = datetime.today().strftime('%Y-%m-%d')
# DATE                = '2026-05-23'
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val' # 'test' or 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 10

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [0.05] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

start = time.time()

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
# TAU_FISTA_UNTRAINED = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
TAU_FISTA_UNTRAINED = [ 1.0 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
# # train and test all possible combinations
# ALPHA       = np.logspace(-3.5, -1.5, 5)
# GAMMA       = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
# CONST_TAU   = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
# PARAMS      = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
# train and test the listed combinations
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


##### TRAINING - TESTING DEEPOPT
ALPHA_DO = [ 0.5 ]
# ALPHA_DO = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
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

    
######################### PRINTING MEMORY USAGE INFO ####################################
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
tf.test.experimental.sync_devices()
print('Elapsed time: ', )
print( f'    {time.time() - start:.2e}s', )
print('GPU memory usage:')
if tf.config.get_visible_devices('GPU') != []:
    print( '    Current:', tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6, 'MB' )
    print( '    Peak:',  tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e6, 'MB' )
else:
    print('GPU not visible')
print('CPU memory usage:')
print( '    Current:', tf.config.experimental.get_memory_info('CPU:0')['current'] / 1e6, 'MB' )
print( '    Peak:',  tf.config.experimental.get_memory_info('CPU:0')['peak'] / 1e6, 'MB' )
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')