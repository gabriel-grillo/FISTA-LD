import os
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors
import pandas as pd

from paths import hist_path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data import get_dataset

######################### PLOTTING OPTIONS ##################################
pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\mean}{mean}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
COLORS = list(mcolors.TABLEAU_COLORS.keys())

#######################################################################################################################
################################################ AUXILIARY FUNCTIONS ##################################################
#######################################################################################################################

def get_label( algorithm, list_of_parameters ):
    if algorithm == 'ista':
        label = 'ISTA'
    elif algorithm == 'fista':
        tau = list_of_parameters
        label = f'FISTA: $\\tau={tau}$'
    elif algorithm == 'fista-ld':
        train_dataset_ratio, miniter, maxiter, alpha, gamma, const_tau, epochs = list_of_parameters
        label = rf'FISTA-LD: N=[{miniter},{maxiter}], '
        label += '$\\alpha=10^{' +  rf'{np.log10(alpha):.1f}' +  '}$, '
        if const_tau <= 0:
            label += rf'$\gamma$={gamma:.3f}, '
        else:
            label += rf'$\tau={const_tau:.2f}$, '
        label += f'train data={train_dataset_ratio}, epochs={epochs}'
    elif algorithm == 'deepopt':
        train_dataset_ratio, miniter, maxiter, alpha, epochs = list_of_parameters
        label = f'DeepOpt: N=[{miniter},{maxiter}], $\\alpha$={alpha:.2f}, '
        label += f'train data={train_dataset_ratio}, epochs={epochs}'
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    return label

def load_data( data_name, test_mode, dataset, test_dataset_ratio, problem, date, fista_list, fista_ld_list, deepopt_list ):
    dicts_list = []
    if data_name in ['Fhist', 'flag_hist', 'inner_iter_hist', 'Rhist']:
        ##### ISTA RESULTS
        if problem != 'lstv':
            hist_path_ista = hist_path( 'ista', [ test_mode, dataset, test_dataset_ratio, problem ] )
            dicts_list.append(
                dict(
                    data  = np.load( os.path.join( hist_path_ista, data_name + '.npy' ) ),
                    label = 'ISTA',
                    ls    = '-.',
                    c     = 'k'
                )
            )
        ##### FISTA RESULTS
        for i,fista_tau in enumerate(fista_list):
            hist_path_fista = hist_path( 'fista', [ test_mode, dataset, test_dataset_ratio, problem, fista_tau ] )
            dicts_list.append(
                dict(
                    data  = np.load( os.path.join( hist_path_fista, data_name + '.npy' ) ),
                    label = get_label( 'fista', fista_tau ),
                    ls    = '-.',
                    c     = COLORS[i%len(COLORS)]
                )
            )
    #### FISTA-LD RESULTS
    for i,options in enumerate(fista_ld_list):
        params, train_dataset_ratio, epochs, minmaxiter = options
        gamma_const_tau, alpha = params
        gamma, const_tau = gamma_const_tau
        miniter, maxiter = minmaxiter
        hist_path_fista_ld = hist_path( 'fista-ld', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio, 
                                                      problem, miniter, maxiter, alpha, gamma, const_tau, date, epochs ] )
        dicts_list.append(
            dict(
                data  = np.load( os.path.join( hist_path_fista_ld, data_name + '.npy') ),
                label = get_label( 'fista-ld', [ train_dataset_ratio, miniter, maxiter, alpha, gamma, const_tau, epochs ] ),
                ls    = '-',
                c     = COLORS[i%len(COLORS)]
            )
        )
    #### DEEPOPT RESULTS
    for i,options in enumerate(deepopt_list):
        alpha, train_dataset_ratio, epochs, minmaxiter = options
        miniter, maxiter = minmaxiter
        hist_path_deepopt = hist_path( 'deepopt', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio,
                                                    problem, miniter, maxiter, alpha, date, epochs ] )
        dicts_list.append(
            dict(
                data  = np.load( os.path.join( hist_path_deepopt, data_name + '.npy') ),
                label = get_label( 'deepopt', [ train_dataset_ratio, miniter, maxiter, alpha, epochs ] ),
                ls    = '--',
                c     = COLORS[i%len(COLORS)]
            )
        )
    return dicts_list

def plot_data( dicts_list, ylabel, fig_path, fig_name ):
    x = dicts_list[0]
    for y in dicts_list[1:]:
        pp.plot( x.get('data'), y.get('data'), label = y.get('label'), ls = y.get('ls'), c = y.get('c') )
    pp.xscale( 'log' )
    pp.yscale( 'log' )
    pp.xlabel( x.get('label') )
    pp.ylabel( ylabel )
    pp.grid()
    pp.legend( loc = 'center left', bbox_to_anchor = ( 1, 0.5 ) )
    pp.savefig( os.path.join( fig_path, fig_name + EXTENSION ) )
    pp.close()

def compute_mean( dicts_list, axis = 0 ):
    mean_dicts_list = [ d.copy() for d in dicts_list ]
    for i,d in enumerate(mean_dicts_list):
        d.update( { 'data':np.mean(dicts_list[i].get('data'),axis=axis) } )
    return mean_dicts_list

def compute_mean_rel_opt_gap( F_dicts ):
    #### COPYING DICTIONARIES
    mean_rel_opt_gap_dicts = [ d.copy() for d in F_dicts ]
    #### COMPUTING F*
    F_hist = np.stack( [ d.get('data') for d in F_dicts ] )
    F_opt = np.min( np.min( F_hist , axis = 2 ), axis = 0 )
    F_opt = F_opt[:,None]
    #### COMPUTING (F-F*)/F*
    rel_opt_gap = ( F_hist - F_opt ) / F_opt
    #### COMPUTING MEAN (F-F*)/F* OVER PROBLEMS
    mean_rel_opt_gap = np.mean( rel_opt_gap, axis = 1 )
    #### CHANGING F BY (F-F*)/F*
    for i,d in enumerate(mean_rel_opt_gap_dicts):
        d.update( { 'data':mean_rel_opt_gap[i] } )
    return mean_rel_opt_gap_dicts

def compute_metrics( test_mode, dataset, test_dataset_ratio, problem, date, fista_ld_list, deepopt_list ):
    val_loss_list = []
    penalty_list = []
    label_list = []
    #### TRAINING DATASET
    ds = get_dataset( dataset, test_mode, problem, batch_size = 1, F_ref = None, overall_ratio = test_dataset_ratio )
    F_train_hist_list = []
    for (_, F_train) in ds:
        F_train_hist_list.append( F_train.numpy()[0] )
    # Reference functional values in training
    F_train_hist = np.stack( F_train_hist_list )
    F_opt_train = np.min( F_train_hist, axis = 1 )[:,None]
    #### FISTA-LD RESULTS
    for options in fista_ld_list:
        params, train_dataset_ratio, epochs, minmaxiter = options
        gamma_const_tau, alpha = params
        gamma, const_tau = gamma_const_tau
        miniter, maxiter = minmaxiter
        hist_path_fista_ld = hist_path( 'fista-ld', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio, 
                                                      problem, miniter, maxiter, alpha, gamma, const_tau, date, epochs ] )
        hist = np.load( os.path.join( hist_path_fista_ld, 'Fhist.npy') )
        val_loss_list.append( np.mean( ( hist[:,miniter:maxiter+1] - F_opt_train ) / ( F_train_hist[:,miniter:maxiter+1] - F_opt_train ) ) )
        penalty_max_iters = min( F_train_hist.shape[1], hist.shape[1] )
        penalty_list.append( np.mean( np.maximum( (hist[:,maxiter+1:penalty_max_iters] - F_train_hist[:,maxiter+1:penalty_max_iters])/F_train_hist[:,maxiter+1:penalty_max_iters], 0.0 ) ) )
        label_list.append( get_label( 'fista-ld', [ train_dataset_ratio, miniter, maxiter, alpha, gamma, const_tau, epochs ] ) )
    #### DEEPOPT RESULTS
    for options in deepopt_list:
        alpha, train_dataset_ratio, epochs, minmaxiter = options
        miniter, maxiter = minmaxiter
        hist_path_deepopt = hist_path( 'deepopt', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio,
                                                    problem, miniter, maxiter, alpha, date, epochs ] )
        hist = np.load( os.path.join( hist_path_deepopt, 'Fhist.npy') )
        val_loss_list.append( np.mean( ( hist[:,miniter:maxiter+1] - F_opt_train ) / ( F_train_hist[:,miniter:maxiter+1] - F_opt_train ) ) )
        penalty_max_iters = min( F_train_hist.shape[1], hist.shape[1] )
        penalty_list.append( np.mean( np.maximum( (hist[:,maxiter+1:penalty_max_iters] - F_train_hist[:,maxiter+1:penalty_max_iters])/F_train_hist[:,maxiter+1:penalty_max_iters], 0.0 ) ) )
        label_list.append( get_label( 'deepopt', [ train_dataset_ratio, miniter, maxiter, alpha, epochs ] ) )
    #### GENERAL ANALYSIS
    val_loss = np.stack( val_loss_list )
    penalty = np.stack( penalty_list )
    return val_loss, penalty, label_list

def print_metrics_analysis( val_loss, penalty, label_list, table_path = None, table_name = None ):
    print('Metrics analysis:')
    print( f'Best validation loss: {np.min(val_loss):.2e}' )
    print( f'Best algorithm w.r.t. validation loss: {label_list[np.argmin(val_loss)]}' )
    print( f'Best penalty: {np.min(penalty):.2e}' )
    print( f'Best parameters w.r.t. penalty: {label_list[np.argmin(penalty)]}' )
    if table_path is not None and table_name is not None:
        table = pd.DataFrame( np.stack( [val_loss, penalty] ).T, index = label_list, columns = [ 'Validation loss', 'Penalty' ] )
        table.index.name = 'Algorithm'
        table.columns.name = 'Metric'
        print( 'Table:' )
        print(table.to_string( float_format = "%.2e", col_space = 12 ), '\n')
        with open( os.path.join( table_path, table_name + '.txt' ), 'w' ) as f:
            f.write(table.to_string( float_format = "%.2e" ) )
            # f.write(table.to_latex( float_format="%.2e" ) )