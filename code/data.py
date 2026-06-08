import os
import numpy as np
import torch
import tensorflow as tf
from pydicom import dcmread
from operators import data_simulation_ray_transform

MAYO_FOLDER = os.path.join( os.getcwd(), 'data', 'mayo_clinic_image', 'full_3mm' )
TEST_RATIO  = 0.2
VAL_RATIO   = 0.1
RANDOM_SEED = 42

def generate_observed_dataset( dataset ):
    assert dataset == 'mayo_clinic_128' or dataset == 'mayo_clinic_512'

    # Numpy random generator
    rng = np.random.default_rng( seed = RANDOM_SEED )

    # Reads files
    data = []
    for (dirpath, dirnames, filenames) in os.walk(MAYO_FOLDER):
        data.extend( [ os.path.join( dirpath, fi ) for fi in filenames ] )
    data = np.array(data)

    if data.size == 0:
        raise ValueError( f'There is no data in {MAYO_FOLDER}. Please see the README.md at {os.path.join( os.getcwd(), 'data', 'mayo_clinic_image' )}.' )

    # Separates dataset into train, test, and validation
    rng.shuffle( data )
    data_size = data.shape[0]
    test_size = int( data_size * TEST_RATIO )
    val_size = int( data_size * VAL_RATIO )
    test_data = data[ :test_size ]
    val_data = data[ test_size : test_size + val_size ]
    train_data = data[ test_size + val_size : ]

    # Creates the transform based on the input dataset
    T = data_simulation_ray_transform( dataset )

    # Generates noisy transformed data
    train_transformed_data = []
    val_transformed_data = []
    test_transformed_data = []
    for data, transformed_data in list( zip( [ train_data, val_data, test_data ], [ train_transformed_data, val_transformed_data, test_transformed_data ] ) ):
        for fn in data:
            # read image from dicom data
            ds = dcmread(fn)
            image = ds.pixel_array
            image = image.astype(np.float32)
            image /= 1000.0
            # generate transformed data with normal 5% noise
            transformed_torch = T( torch.from_numpy(image).float().cuda() )
            transformed = transformed_torch.detach().cpu().numpy()
            noisy = transformed + 0.05 * np.mean( np.abs( transformed ) ) * rng.normal( size = transformed.shape ).astype( np.float32 )
            transformed_data.append( noisy[...,None] )

    # Saves Datasets
    dataset_path = os.path.join( os.getcwd(), 'data', dataset, 'observed' )
    tf.data.Dataset.save( tf.data.Dataset.from_tensor_slices( train_transformed_data ), os.path.join( dataset_path, 'train' ) )
    tf.data.Dataset.save( tf.data.Dataset.from_tensor_slices(   val_transformed_data ), os.path.join( dataset_path,   'val' ) )
    tf.data.Dataset.save( tf.data.Dataset.from_tensor_slices(  test_transformed_data ), os.path.join( dataset_path,  'test' ) )
        
def generate_F_dataset( dataset, problem, F_ref ):
    dataset_path = os.path.join( os.getcwd(), 'data', dataset )
    for mode in ['train', 'val', 'test']:
        # Loading the observed sinograms dataset
        obs_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'observed', mode ) )
        # Operations in batches of size 32.
        obs_dataset = obs_dataset.batch( 32 )
        # Empty list of functional values
        F_list = []
        # Running through examples
        for b in obs_dataset:
            hist = F_ref( b )
            F_list.append( hist )
        # Concatenation of the history
        overall_hist = tf.concat( F_list, axis = 0 )
        # Saves the functional values for reference dataset
        tf.data.Dataset.save( tf.data.Dataset.from_tensor_slices( overall_hist ), \
                              os.path.join( dataset_path, 'F_ref', problem, mode ) )

def get_dataset( dataset, mode, problem, batch_size, F_ref, overall_ratio = 1.0 ):
    assert mode == 'train' or mode == 'val' or mode == 'test'

    # Checks whether the datasets hasn't been generated yet
    dataset_path = os.path.join( os.getcwd(), 'data', dataset )
    if not os.path.exists( os.path.join( dataset_path, 'observed', mode ) ):
        # Generates observed sinograms and saves them
        print('Generating the simulated sinograms...')
        generate_observed_dataset( dataset )
        print('Done!')
    if not os.path.exists( os.path.join( dataset_path, 'F_ref', problem, mode ) ):
        # Generates the functional values for reference
        print('Generating the functional values for reference...')
        generate_F_dataset( dataset, problem, F_ref )
        print('Done!')

    # Loads dataset
    obs_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'observed', mode ) )
    F_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'F_ref', problem, mode ) )

    # Zips both datasets together
    Dataset = tf.data.Dataset.zip( obs_dataset, F_dataset )

    # Takes a fraction of the dataset
    examples_to_take = int( Dataset.cardinality().numpy() * overall_ratio )
    Dataset = Dataset.take( examples_to_take )

    if mode == 'train':
        # Shuffles train examples at the beginning of each epoch
        Dataset = Dataset.shuffle( 500, reshuffle_each_iteration = True, seed = RANDOM_SEED )

    # Combines examples on batches
    Dataset = Dataset.batch( batch_size )

    return Dataset
