import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors
from itertools import product
import warnings
import pandas as pd

COLORS = list(mcolors.TABLEAU_COLORS.keys())
FISTA_COLORS = ['dimgray', 'navy', 'saddlebrown'] # maybe three are enough

######################### PLOTTING OPTIONS ##################################
# pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
# pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\sen}{sen}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
# SAVEDATE = '2026-05-13'
SAVEDATE = datetime.today().strftime('%Y-%m-%d')

######################### RESULTS PARAMETERS ##################################
MODE                = 'val'
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lasso' # 'nnls' or 'lasso' or 'slasso' or 'nnslasso'
EPOCHS              = 20
DATE                = '2026-03-30'
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAINED_ITERS       = list( product( MINITER, MAXITER ) )
PLOT_ITERS          = 1000

TRAIN_DATASET_RATIO = 0.05
TEST_DATASET_RATIO  = 1.0

### FISTA parameters
TAU_UNTRAINED_FISTA = [1.0]

### FISTA-LD parameters
ALPHA               = np.logspace( -3.5, -1.5, 5 )
GAMMA               = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
ALPHA_GAMMA         = list( product( ALPHA, GAMMA ) )

######################### COMPUTING F* ##################################
#### FISTA-LD RESULTS
F_fistald_all = []
for options in product( ALPHA_GAMMA, TRAINED_ITERS ):
    alpha_gamma, minmaxiter = options
    alpha, gamma = alpha_gamma
    miniter, maxiter = minmaxiter

    result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'fista-ld', DATASET, \
                                f'{100*TRAIN_DATASET_RATIO}percent_train', \
                                f'{100*TEST_DATASET_RATIO}percent_test', \
                                PROBLEM )
    if maxiter == miniter:
        result_path = os.path.join( result_path, 'iters_fixed_' + str(maxiter) )
    elif maxiter > miniter:
        result_path = os.path.join( result_path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:] )
    result_path = os.path.join( result_path, 'date_' + DATE )
    result_path = os.path.join( result_path, f'epoch_{EPOCHS}' )
    res_filename = os.path.join( result_path, 'Fhist.npy')
    if os.path.exists(res_filename):
        F_fistald_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
F_fistald_all = np.stack( F_fistald_all )

Ffinal = np.min( np.min(F_fistald_all, axis = 2), axis = 0 )
##### FISTA RESULTS
Ffista_all = []
for fista_tau in TAU_UNTRAINED_FISTA:
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
    if os.path.exists(res_filename):
        Ffista_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
Ffista_all = np.stack( Ffista_all )
##### COMPARING FISTA AND FISTA-LD
Ffinal = np.minimum( Ffinal, np.min( np.min( Ffista_all, axis = 2), axis = 0 ) )
##### ISTA RESULTS
res_filename = os.path.join( 'test_results', MODE, 'untrained', 'ista', DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
if os.path.exists(res_filename):
    F_ista = np.load( res_filename )
else:
    raise ValueError(res_filename)
##### COMPARING ISTA, FISTA, AND FISTA-LD
Ffinal = np.minimum( Ffinal, np.min( F_ista, axis = 1 ) )
Ffinal = Ffinal[:,None]

######################### UNTRAINED FISTA AND ISTA RESULTS - RELATIVE OPT GAP ##################################
for i,_ in enumerate(TAU_UNTRAINED_FISTA):
    Ffista_all[i] -= Ffinal
    Ffista_all[i] /= Ffinal
F_ista -= Ffinal
F_ista /= Ffinal
F_ista_mean = np.mean( F_ista, axis = 0 )[:PLOT_ITERS+1]
F_ista_std = np.std( F_ista, axis = 0 )[:PLOT_ITERS+1]

######################### CHECKING ISTA'S MONOTONY ##################################
is_nonmonotone = np.any( F_ista[:,1:] > F_ista[:,:-1], axis = 1 )
if np.sum( is_nonmonotone ) > 0:
    warnings.warn( f'ISTA has a non-monotonic step in {np.sum( is_nonmonotone )} out of {F_ista.shape[0]} problems.' )

######################### COMPUTING ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss_matrix = np.zeros( (len(ALPHA), len(GAMMA)) )
penalty_matrix = np.zeros( (len(ALPHA), len(GAMMA)) )
miniter, maxiter = 1, 20
fista_tau = 1.0 # comparing with FISTA tau = 1.0
for i, alpha in enumerate( ALPHA ):
    for j, gamma in enumerate( GAMMA ):
        result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'fista-ld', DATASET, \
                                    f'{100*TRAIN_DATASET_RATIO}percent_train', \
                                    f'{100*TEST_DATASET_RATIO}percent_test', \
                                    PROBLEM )
        if maxiter == miniter:
            result_path = os.path.join( result_path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            result_path = os.path.join( result_path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:] )
        result_path = os.path.join( result_path, 'date_' + DATE )
        result_path = os.path.join( result_path, f'epoch_{EPOCHS}' )
        res_filename_FISTALD = os.path.join( result_path, 'Fhist.npy')
        res_filename_FISTA = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
        if os.path.exists(res_filename_FISTA) and os.path.exists(res_filename_FISTALD):
            # Computing validation loss
            FfinalFISTA = np.load( res_filename_FISTA )[:,-1]
            FfinalFISTA = FfinalFISTA[:,None]
            FhistFISTA = np.load( res_filename_FISTA )[:,miniter:maxiter+1]
            FhistFISTALD = np.load( res_filename_FISTALD )[:,miniter:maxiter+1]
            val_loss_matrix[i,j] = np.mean( ( FhistFISTALD - FfinalFISTA ) / ( FhistFISTA - FfinalFISTA ) )
            # Computing penalty
            FhistFISTA = np.load( res_filename_FISTA )[:,maxiter+1:PLOT_ITERS+1]
            FhistFISTALD = np.load( res_filename_FISTALD )[:,maxiter+1:PLOT_ITERS+1]
            penalty_matrix[i,j] = np.mean( np.maximum( (FhistFISTALD - FhistFISTA)/FhistFISTA, 0.0 ) )
# Computing best parameters w.r.t. validation loss and penalty
i_alpha_best_val_loss, j_gamma_best_val_loss = np.unravel_index(np.argmin(val_loss_matrix, axis=None), val_loss_matrix.shape)
alpha_best_val_loss = ALPHA[i_alpha_best_val_loss]
gamma_best_val_loss = GAMMA[j_gamma_best_val_loss]
print( f'Best validation loss: {np.min(val_loss_matrix):.2e}' )
print( f'Best parameters w.r.t. validation loss: alpha = {alpha_best_val_loss:.2e} and gamma = {gamma_best_val_loss:.2e}' )
print( 'Validation loss matrix:' )
tabela_val_loss = pd.DataFrame( val_loss_matrix, index = [ f'{a:.2e}' for a in ALPHA], columns = [ f'{a:.2e}' for a in GAMMA] )
tabela_val_loss.index.name = 'alpha'
tabela_val_loss.columns.name = 'gamma'
print(tabela_val_loss.to_string( float_format="%.2e", col_space=12 ), '\n')
# print(tabela_val_loss.to_latex( float_format="%.2e" ), '\n')
i_alpha_best_penalty, j_gamma_best_penalty = np.unravel_index(np.argmin(penalty_matrix, axis=None), penalty_matrix.shape)
alpha_best_penalty = ALPHA[i_alpha_best_penalty]
gamma_best_penalty = GAMMA[j_gamma_best_penalty]
print( f'Best penalty: {np.min(penalty_matrix):.2e}' )
print( f'Best parameters w.r.t. penalty: alpha = {alpha_best_penalty:.2e} and gamma = {gamma_best_penalty:.2e}' )
print( 'Penalty matrix:' )
tabela_penalty = pd.DataFrame( penalty_matrix, index = [ f'{a:.2e}' for a in ALPHA], columns = [ f'{a:.2e}' for a in GAMMA]  )
tabela_penalty.index.name = 'alpha'
tabela_penalty.columns.name = 'gamma'
print(tabela_penalty.to_string( float_format="%.2e", col_space=12 ), '\n')
# print(tabela_penalty.to_latex( float_format="%.2e" ), '\n')

print('Start ploting ...')
######################### ANALYSING THE VARIATIONS GAMMA - FIXING ALPHA ##################################
### SAVING ISTA AND FISTA DATA
data_alphafixed_label = [ 'k+1', 'ISTA' ]
data_alphafixed = [1 + np.arange(PLOT_ITERS+1), F_ista_mean]
data_alphafixed_std = [1 + np.arange(PLOT_ITERS+1), F_ista_std]
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_alphafixed_label.append( f'FISTA tau={fista_tau}' )
    data_alphafixed.append( Fhist_mean )
    data_alphafixed_std.append( Fhist_std )
### ITERATES OVER ALPHA
for alpha in ALPHA:
    list_of_param_comb = list( product( list(zip(MINITER,MAXITER)), GAMMA ) )
    trained_algorithms = len( list_of_param_comb )
    ######################### TESTING INFO PATH ##################################
    RESULT_PATH = []
    for options in list_of_param_comb:
        minmaxiter, gamma = options
        miniter, maxiter = minmaxiter

        result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'fista-ld', DATASET, \
                                            f'{100*TRAIN_DATASET_RATIO}percent_train', \
                                            f'{100*TEST_DATASET_RATIO}percent_test', \
                                            PROBLEM )
        if maxiter == miniter:
            result_path = os.path.join( result_path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            result_path = os.path.join( result_path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:] )
        result_path = os.path.join( result_path, 'date_' + DATE )
        result_path = os.path.join( result_path, f'epoch_{EPOCHS}' )

        RESULT_PATH.append(result_path)

    ######################### SETTING FIGURES' PATH ##################################
    FIG_PATH = os.path.join(os.getcwd(), 'plots', DATASET, PROBLEM, SAVEDATE, 'alpha_fixed' )
    if not os.path.exists(FIG_PATH):
        os.makedirs(FIG_PATH)

    ######################### CREATING LABELS FOR FISTA-LD ##################################
    labels_fista_ld = []
    for i,options in enumerate(list_of_param_comb):
        minmaxiter, gamma = options
        miniter, maxiter = minmaxiter
        label = 'FISTA-LD: '
        label += f'$\\alpha=10^{np.log10(alpha):.1f}$, '
        label += f'$\\gamma$={gamma:.2e}.'
        labels_fista_ld.append( label )
        # print( str(i+1) + '  ' + label )
    ##### INSERT HERE THE DESIRED FISTA-LD TO PLOT
    fista_ld_to_plot = range(trained_algorithms)

    ######################### PLOTTING MEAN RELATIVE OPT GAP -- ITERS ##################################
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  F_ista_mean, label = 'ISTA', c = 'k', ls = '-.' )
    pp.fill_between( 1 + np.arange(PLOT_ITERS+1), F_ista_mean - F_ista_std, F_ista_mean + F_ista_std, color = 'k', alpha = 0.5 )
    for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
        Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
        Fhist_mean = np.mean( Fhist, axis = 0 )
        Fhist_std = np.std( Fhist, axis = 0 )
        pp.plot( 1 + np.arange(PLOT_ITERS+1), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i] )
        pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = FISTA_COLORS[i], alpha = 0.5 )
    for i in range(trained_algorithms):
        res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
        if os.path.exists(res_filename):
            if i in fista_ld_to_plot:
                Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
                Fhist -= Ffinal
                Fhist /= Ffinal
                Fhist_mean = np.mean( Fhist, axis = 0 )
                Fhist_std = np.std( Fhist, axis = 0 )
                data_alphafixed_label.append( labels_fista_ld[i] )
                pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = labels_fista_ld[i], c = COLORS[i] )
                data_alphafixed.append( Fhist_mean )
                pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i], alpha = 0.5 )
                data_alphafixed_std.append( Fhist_std )
        else:
            raise ValueError()
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel('$k+1$')
    pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
    pp.grid()
    pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig( os.path.join( FIG_PATH, 'optgap' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( TEST_DATASET_RATIO * 100 ) + 'percent' + '_alpha_' + f'alpha={alpha:.2e}' + EXTENSION ) )
    pp.close()

    ######################### PLOTTING MEAN R_k ##################################
    pp.loglog( 1 + np.arange(PLOT_ITERS+1),  1e+6/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k' )
    pp.loglog( 1 + np.arange(9,PLOT_ITERS+1),  1e+9/(1 + np.arange(9,PLOT_ITERS+1))**3, label = '$O(1/k^3)$', c = 'k', ls = '--' )
    for i in range(trained_algorithms):
        res_filename = os.path.join( RESULT_PATH[i], 'Rhist.npy')
        if not os.path.exists(res_filename):
            raise ValueError()
        
        Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
        Rhist_mean = np.mean( Rhist, axis = 0 )
        Rhist_std = np.std( Rhist, axis = 0 )
        pp.plot( 1 + np.arange(Rhist.shape[1]), Rhist_mean, label = labels_fista_ld[i], c = COLORS[i] )
        pp.fill_between( 1 + np.arange(Rhist.shape[1]), Rhist_mean - Rhist_std, Rhist_mean + Rhist_std, color = COLORS[i], alpha = 0.5 )
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel('$k+1$')
    pp.ylabel('mean$\\left(R_k\\right)$')
    pp.grid()
    pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig( os.path.join( FIG_PATH, 'meanR' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( TEST_DATASET_RATIO * 100 ) + 'percent' + '_alpha_' + f'alpha={alpha:.2e}' + EXTENSION ) )
    pp.close()


######################### ANALYSING THE VARIATIONS ALPHA - FIXING GAMMA ##################################
### SAVING ISTA AND FISTA DATA
data_gammafixed_label = [ 'k+1', 'ISTA' ]
data_gammafixed = [1 + np.arange(PLOT_ITERS+1), F_ista_mean]
data_gammafixed_std = [1 + np.arange(PLOT_ITERS+1), F_ista_std]
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_gammafixed_label.append( f'FISTA tau={fista_tau}' )
    data_gammafixed.append( Fhist_mean )
    data_gammafixed_std.append( Fhist_std )
### ITERATES OVER GAMMA
for gamma in GAMMA:
    list_of_param_comb = list( product( list(zip(MINITER,MAXITER)), ALPHA ) )
    trained_algorithms = len( list_of_param_comb )

    ######################### TESTING INFO PATH ##################################
    RESULT_PATH = []
    for options in list_of_param_comb:
        minmaxiter, alpha = options
        miniter, maxiter = minmaxiter

        result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'fista-ld', DATASET, \
                                            f'{100*TRAIN_DATASET_RATIO}percent_train', \
                                            f'{100*TEST_DATASET_RATIO}percent_test', \
                                            PROBLEM )
        if maxiter == miniter:
            result_path = os.path.join( result_path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            result_path = os.path.join( result_path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:] )
        result_path = os.path.join( result_path, 'date_' + DATE )
        result_path = os.path.join( result_path, f'epoch_{EPOCHS}' )

        RESULT_PATH.append(result_path)

    ######################### SETTING FIGURES' PATH ##################################
    FIG_PATH = os.path.join(os.getcwd(), 'plots', DATASET, PROBLEM, SAVEDATE, 'gamma_fixed' )
    if not os.path.exists(FIG_PATH):
        os.makedirs(FIG_PATH)

    ######################### CREATING LABELS FOR FISTA-LD ##################################
    labels_fista_ld = []
    for i,options in enumerate(list_of_param_comb):
        minmaxiter, alpha = options
        miniter, maxiter = minmaxiter
        label = 'FISTA-LD: '
        label += f'$\\alpha=10^{np.log10(alpha):.1f}$, '
        label += f'$\\gamma=${gamma:.2e}.'
        labels_fista_ld.append( label )
        # print( str(i+1) + '  ' + label )
    ##### INSERT HERE THE DESIRED FISTA-LD TO PLOT
    fista_ld_to_plot = range(trained_algorithms)

    ######################### PLOTTING MEAN RELATIVE OPT GAP -- ITERS ##################################
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  F_ista_mean, label = 'ISTA', c = 'k', ls = '-.' )
    pp.fill_between( 1 + np.arange(PLOT_ITERS+1), F_ista_mean - F_ista_std, F_ista_mean + F_ista_std, color = 'k', alpha = 0.5 )
    for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
        Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
        Fhist_mean = np.mean( Fhist, axis = 0 )
        Fhist_std = np.std( Fhist, axis = 0 )
        pp.plot( 1 + np.arange(PLOT_ITERS+1), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i] )
        pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = FISTA_COLORS[i], alpha = 0.5 )
    for i in range(trained_algorithms):
        res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
        if os.path.exists(res_filename):
            if i in fista_ld_to_plot:
                Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
                Fhist -= Ffinal
                Fhist /= Ffinal
                Fhist_mean = np.mean( Fhist, axis = 0 )
                Fhist_std = np.std( Fhist, axis = 0 )
                data_gammafixed_label.append( labels_fista_ld[i] )
                pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = labels_fista_ld[i], c = COLORS[i] )
                data_gammafixed.append( Fhist_mean )
                pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i], alpha = 0.5 )
                data_gammafixed_std.append( Fhist_std )
        else:
            raise ValueError()
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel('$k+1$')
    pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
    # pp.axhline( np.finfo(np.float32).eps, c = 'blue', alpha = 0.4, label = '$\\epsilon_{\\text{float32}}$' )
    pp.grid()
    pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig( os.path.join( FIG_PATH, 'optgap' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( TEST_DATASET_RATIO * 100 ) + 'percent' + '_gamma_' + f'{gamma:.2e}' + EXTENSION ) )
    pp.close()

    ######################### PLOTTING MEAN R_k ##################################
    pp.loglog( 1 + np.arange(PLOT_ITERS+1),  1e+6/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k' )
    pp.loglog( 1 + np.arange(9,PLOT_ITERS+1),  1e+9/(1 + np.arange(9,PLOT_ITERS+1))**3, label = '$O(1/k^3)$', c = 'k', ls = '--' )
    for i in range(trained_algorithms):
        res_filename = os.path.join( RESULT_PATH[i], 'Rhist.npy')
        if not os.path.exists(res_filename):
            raise ValueError()
        
        Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
        Rhist_mean = np.mean( Rhist, axis = 0 )
        Rhist_std = np.std( Rhist, axis = 0 )
        pp.plot( 1 + np.arange(Rhist.shape[1]), Rhist_mean, label = labels_fista_ld[i], c = COLORS[i] )
        pp.fill_between( 1 + np.arange(Rhist.shape[1]), Rhist_mean - Rhist_std, Rhist_mean + Rhist_std, color = COLORS[i], alpha = 0.5 )
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel('$k+1$')
    pp.ylabel('mean$\\left(R_k\\right)$')
    pp.grid()
    pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig( os.path.join( FIG_PATH, 'meanR' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( TEST_DATASET_RATIO * 100 ) + 'percent' + '_gamma_' + f'{gamma:.2e}' + EXTENSION ) )
    pp.close()
print('Done!')
