import os
from datetime import datetime
from math import log10, floor
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors
from itertools import product
import warnings
import pandas as pd

######################### PLOTTING OPTIONS ##################################
# pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
# pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\sen}{sen}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
COLORS = list(mcolors.TABLEAU_COLORS.keys())

######################### RESULTS PARAMETERS ##################################
MODE                = 'val'
DATASET             = 'mayo_clinic_128'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lasso'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso' or 'lstv'
EPOCHS_TO_PLOT      = [ 20 ]
DATE                = datetime.today().strftime('%Y-%m-%d')
# DATE                = '2026-05-31'
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAINED_ITERS       = list(zip(MINITER,MAXITER))
PLOT_ITERS          = 1000
PLOT_ISTA           = True

TRAIN_DATASET_RATIO = [0.05]
TEST_DATASET_RATIO  = 0.05

### FISTA parameters
# TAU_UNTRAINED_FISTA = [0.1,0.25, 0.5, 0.75,0.9]
TAU_UNTRAINED_FISTA = [1.0]

### FISTA-LD parameters
# # analyse all possible combinations
# ALPHA           = np.logspace(-3.5, -1.5, 5)
# GAMMA           = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
# CONST_TAU       = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
# FISTA_LD_PARAMS = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
# analyse the listed combinations
ALPHA           = [ 10**(-2.5) ]
GAMMA           = [ 0.05 ]
CONST_TAU       = [ -1.0 ]
FISTA_LD_PARAMS = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
trained_fista_ld = len( list( product( FISTA_LD_PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, TRAINED_ITERS ) ) )

### DeepOpt parameters
PLOT_DEEPOPT = True
if PLOT_DEEPOPT:
    MINITER_DO = [ 1 ]
    MAXITER_DO = [ 20 ]
    # ALPHA_DO   = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
    ALPHA_DO   = [ 0.5 ]
else:
    MINITER_DO = []
    MAXITER_DO = []
    ALPHA_DO   = []
DATE_DO    = datetime.today().strftime('%Y-%m-%d')
# DATE_DO    = '2026-05-31'
trained_deepopt = len( list( product( ALPHA_DO, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, list(zip(MINITER_DO,MAXITER_DO)) ) ) )

######################### SETTING FIGURES' PATH ##################################
# SAVEDATE = '2026-05-23'
SAVEDATE = datetime.today().strftime('%Y-%m-%d')
FIG_PATH = os.path.join(os.getcwd(), 'plots', DATASET, PROBLEM, SAVEDATE )
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


######################### TESTING INFO PATH ##################################
RESULT_PATH = []
for options in product( FISTA_LD_PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, TRAINED_ITERS ):
    params, train_dataset_ratio, epochs, minmaxiter = options
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = minmaxiter

    result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'fista-ld', DATASET, \
                                f'{100*train_dataset_ratio}percent_train', \
                                f'{100*TEST_DATASET_RATIO}percent_test', \
                                PROBLEM )
    if maxiter == miniter:
        result_path = os.path.join( result_path, 'iters_fixed_' + str(maxiter) )
    elif maxiter > miniter:
        result_path = os.path.join( result_path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    if const_tau <= 0:
        result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'gamma' + str(gamma)[2:]  )
    else:
        result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:], 'const_tau' + str(const_tau)[2:]  )
    result_path = os.path.join( result_path, 'date_' + DATE )
    result_path = os.path.join( result_path, f'epoch_{epochs}' )

    RESULT_PATH.append(result_path)


######################### COMPUTING F* ##################################
#### FISTA-LD RESULTS
F_fistald_all = []
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if os.path.exists(res_filename):
        F_fistald_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
if len(F_fistald_all) > 0:
    Ffinal = np.min( np.stack( [ np.min(hist, axis=1) for hist in F_fistald_all ] ), axis = 0 )
else:
    Ffinal = np.inf
##### FISTA RESULTS
Ffista_all = []
for fista_tau in TAU_UNTRAINED_FISTA:
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
    if os.path.exists(res_filename):
        Ffista_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
##### COMPARING FISTA AND FISTA-LD
if len(Ffista_all) > 0:
    Ffinal = np.minimum( Ffinal, np.min( np.stack( [ np.min(hist, axis=1) for hist in Ffista_all ] ), axis = 0 ) )
##### ISTA RESULTS
if PLOT_ISTA:
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
if PLOT_ISTA:
    F_ista -= Ffinal
    F_ista /= Ffinal
    F_ista_mean = np.mean( F_ista, axis = 0 )[:PLOT_ITERS+1]
    F_ista_std = np.std( F_ista, axis = 0 )[:PLOT_ITERS+1]

######################### CHECKING ISTA'S MONOTONY ##################################
if PLOT_ISTA:
    is_nonmonotone = np.any( F_ista[:,1:] > F_ista[:,:-1], axis = 1 )
    if np.sum( is_nonmonotone ) > 0:
        warnings.warn( f'ISTA has a non-monotonic step in {np.sum( is_nonmonotone )} out of {F_ista.shape[0]} problems.' )

######################### CREATING LABELS FOR FISTA-LD ##################################
print('\n---------------------------------------------------------------- Printing the labels (FISTA-LD) --------------------------------------------------------------------')
labels_fista_ld = []
for options in product( FISTA_LD_PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, TRAINED_ITERS ):
    params, train_dataset_ratio, epochs, minmaxiter = options
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = minmaxiter
    label = 'FISTA-LD: '
    label += f'N=[{miniter},{maxiter}], '
    label += f'$\\alpha$={alpha:.2e}, '
    if const_tau <= 0:
        label += f'$\\gamma$={gamma:.2e}, '
    else:
        label += f'$\\tau={const_tau:.2f}$, '
    label += f'train data={train_dataset_ratio}, '
    label += f'epochs={epochs}'
    labels_fista_ld.append( label )
    print( label )
print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
##### INSERT HERE THE DESIRED FISTA-LD TO PLOT
fista_ld_to_plot = range(trained_fista_ld)
# fista_ld_to_plot = [1,2,4,5]

######################### PLOTTING MEAN RELATIVE OPT GAP ##################################
data = [ 1 + np.arange(PLOT_ITERS+1) ]
data_std = [ 1 + np.arange(PLOT_ITERS+1) ]
data_label = [ 'k+1' ]
pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
if PLOT_ISTA:
    data_label.append( 'ISTA' )
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  F_ista_mean, label = 'ISTA', c = 'k', ls = '-.' )
    data.append( F_ista_mean )
    pp.fill_between( 1 + np.arange(PLOT_ITERS+1), F_ista_mean - F_ista_std, F_ista_mean + F_ista_std, color = 'k', alpha = 0.5 )
    data_std.append( F_ista_std )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( 'FISTA $\\tau={fista_tau}$' )
    pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i % len(COLORS)] )
    data.append( Fhist_mean )
    pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.5 )
    data_std.append( Fhist_std )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if os.path.exists(res_filename):
        if i in fista_ld_to_plot:
            Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
            Fhist -= Ffinal
            Fhist /= Ffinal
            Fhist_mean = np.mean( Fhist, axis = 0 )
            Fhist_std = np.std( Fhist, axis = 0 )
            data_label.append( labels_fista_ld[i] )
            pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = labels_fista_ld[i] )
            data.append( Fhist_mean )
            # pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, alpha = 0.5 )
            data_std.append( Fhist_std )
    else:
        raise ValueError()
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
with open( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")

######################### PLOTTING MEAN R_k ##################################
data = [ 1 + np.arange(PLOT_ITERS+1) ]
data_std = [ 1 + np.arange(PLOT_ITERS+1) ]
data_label = [ 'k+1' ]
pp.plot( 1 + np.arange(PLOT_ITERS+1),  1e+6/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k' )
pp.plot( 1 + np.arange(9,PLOT_ITERS+1),  1e+9/(1 + np.arange(9,PLOT_ITERS+1))**3, label = '$O(1/k^3)$', c = 'k', ls = '--' )
if PROBLEM == 'lstv':
    for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
        res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Rhist.npy' )
        if not os.path.exists(res_filename):
            raise ValueError()
        Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
        Rhist_mean = np.mean( Rhist, axis = 0 )
        Rhist_std = np.std( Rhist, axis = 0 )
        data_label.append( 'FISTA $\\tau={fista_tau}$' )
        pp.plot( 1 + np.arange(Rhist.shape[1]), Rhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i % len(COLORS)] )
        data.append( Rhist_mean )
        pp.fill_between( 1 + np.arange(Rhist.shape[1]), Rhist_mean - Rhist_std, Rhist_mean + Rhist_std, color = COLORS[i % len(COLORS)], alpha = 0.5 )
        data_std.append( Rhist_std )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Rhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Rhist_mean = np.mean( Rhist, axis = 0 )
    Rhist_std = np.std( Rhist, axis = 0 )
    data_label.append( labels_fista_ld[i] )
    pp.plot( 1 + np.arange(Rhist.shape[1]), Rhist_mean, label = labels_fista_ld[i], c = COLORS[i % len(COLORS)] )
    data.append( Rhist_mean )
    pp.fill_between( 1 + np.arange(Rhist.shape[1]), Rhist_mean - Rhist_std, Rhist_mean + Rhist_std, color = COLORS[i % len(COLORS)], alpha = 0.5 )
    data_std.append( Rhist_std )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(R_k\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'meanR_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'meanR_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'meanR_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
with open( os.path.join( FIG_PATH, 'meanR_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")

#############################################################################################
######################### COMPARING WITH DEEP OPTIMIZATION ##################################
#############################################################################################

######################### TESTING INFO PATH ##################################
RESULT_PATH_DO = []
for options in product( ALPHA_DO, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, list(zip(MINITER_DO,MAXITER_DO)) ):
    alpha, train_dataset_ratio, epochs, minmaxiter = options
    miniter, maxiter = minmaxiter
    result_path = os.path.join( os.getcwd(), 'test_results', MODE, 'trained', 'deepopt', DATASET,\
                                f'{100*train_dataset_ratio}percent_train', \
                                f'{100*TEST_DATASET_RATIO}percent_test', \
                                PROBLEM )
    if maxiter == miniter:
        trained_iterations = 'fixed'
        result_path = os.path.join( result_path, 'iters_' + trained_iterations + '_' + str(maxiter) )
    elif maxiter > miniter:
        trained_iterations = 'random'
        result_path = os.path.join( result_path, 'iters_' + trained_iterations + '_' + str(miniter) + '_' + str(maxiter) ) 
    else:
        raise ValueError()
    result_path = os.path.join( result_path, 'alpha' + str(alpha)[2:] )
    result_path = os.path.join( result_path, 'date_' + DATE_DO )
    result_path = os.path.join( result_path, f'epoch_{epochs}' )

    RESULT_PATH_DO.append( result_path )

######################### RECOMPUTING F* CONSIDERING DEEPOPT ##################################
##### DEEPOPT RESULTS
if PLOT_DEEPOPT:
    Fall_list = []
    for i in range(trained_deepopt):
        res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy')
        if not os.path.exists(res_filename):
            raise ValueError(res_filename)
        Fall_list.append( np.load( res_filename ) )
    Fall = np.stack( Fall_list )
    Fdeepopt = np.min( np.min(Fall, axis = 2), axis = 0 )
    ##### COMPARING DEEPOPT AND PREVIOUS (FISTA AND FISTA-LD) RESULTS
    Ffinal = np.minimum( Ffinal, Fdeepopt[:,None] )

######################### UNTRAINED FISTA AND ISTA RESULTS - RELATIVE OPT GAP ##################################
Ffista_all = []
for fista_tau in TAU_UNTRAINED_FISTA:
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
    if os.path.exists(res_filename):
        Ffista_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
Ffista_all = np.stack( Ffista_all )
for i,_ in enumerate(TAU_UNTRAINED_FISTA):
    Ffista_all[i] -= Ffinal
    Ffista_all[i] /= Ffinal
if PLOT_ISTA:
    F_ista = np.load( os.path.join( 'test_results', MODE, 'untrained', 'ista', DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' ) )
    F_ista -= Ffinal
    F_ista /= Ffinal
    F_ista_mean = np.mean( F_ista, axis = 0 )[:PLOT_ITERS+1]
    F_ista_std = np.std( F_ista, axis = 0 )[:PLOT_ITERS+1]

######################### CREATING LABELS FOR DEEPOPT ##################################
print('\n---------------------------------------------------------------- Printing the labels (DeepOpt) --------------------------------------------------------------------')
labels_deepopt = []
for options in product( ALPHA_DO, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, list(zip(MINITER_DO,MAXITER_DO)) ):
    alpha, train_dataset_ratio, epochs, minmaxiter = options
    miniter, maxiter = minmaxiter
    label = 'DeepOpt: '
    label += f'N=[{miniter},{maxiter}], '
    label += f'$\\alpha$={alpha}, '
    label += f'train data={train_dataset_ratio}, '
    label += f'epochs={epochs}'
    labels_deepopt.append( label )
    print( label )
print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
##### INSERT HERE THE DESIRED DEEPOPT TO PLOT
deepopt_to_plot = range( trained_deepopt )


######################### PLOTTING MEAN FUNCTIONAL RELATIVE ERROR -- only trained iterations ##################################
if PLOT_DEEPOPT:
    maxiter_all = max( max(MAXITER), max(MAXITER_DO) )
else:
    maxiter_all = max(MAXITER)
data = [ 1 + np.arange(maxiter_all+1) ]
data_std = [ 1 + np.arange(maxiter_all+1) ]
data_label = [ 'k+1' ]
pp.semilogy( np.arange(maxiter_all+1),  6000/(1 + np.arange(maxiter_all+1))**2, label = '$O(1/k^2)$', c = 'k', ls = ':' )
if PLOT_ISTA:
    data_label.append( 'ISTA' )
    pp.plot( np.arange(maxiter_all+1),  F_ista_mean[:maxiter_all+1], label = 'ISTA', c = 'k', ls = '-.' )
    data.append( F_ista_mean[:maxiter_all+1] )
    pp.fill_between( np.arange(maxiter_all+1), F_ista_mean[:maxiter_all+1] - F_ista_std[:maxiter_all+1], F_ista_mean[:maxiter_all+1] + F_ista_std[:maxiter_all+1], color = 'k', alpha = 0.3 )
    data_std.append( F_ista_std[:maxiter_all+1] )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:maxiter_all+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( f'FISTA $\\tau={fista_tau}$' )
    pp.plot( np.arange(maxiter_all+1), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i % len(COLORS)] )
    data.append( Fhist_mean )
    pp.fill_between( np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
    data_std.append( Fhist_std )
##### LEARNED FISTA
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:maxiter_all+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( labels_fista_ld[i] )
    pp.plot( np.arange(Fhist.shape[1]), Fhist_mean, label = labels_fista_ld[i], c = COLORS[i % len(COLORS)] )
    data.append( Fhist_mean )
    pp.fill_between( np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
    data_std.append( Fhist_std )
##### DEEP OPTIMZATION
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy' )
    if not os.path.exists(res_filename):
        print(res_filename)
        raise ValueError()
    Fhist = np.load( res_filename )[:,:maxiter_all+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( labels_deepopt[i] )
    pp.plot( np.arange(Fhist.shape[1]), Fhist_mean, label = labels_deepopt[i], c = COLORS[i % len(COLORS)], ls = '--' )
    data.append( Fhist_mean )
    pp.fill_between( np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
    data_std.append( Fhist_std )
pp.yscale('log')
pp.xlabel('$k$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.xticks(np.arange(0,maxiter_all+1,2))
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'do_zoom_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'do_zoom_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'do_zoom_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
with open( os.path.join( FIG_PATH, 'do_zoom_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")


######################### PLOTTING MEAN FUNCTIONAL RELATIVE ERROR ##################################
data = [ 1 + np.arange(PLOT_ITERS+1) ]
data_std = [ 1 + np.arange(PLOT_ITERS+1) ]
data_label = [ 'k+1' ]
pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
if PLOT_ISTA:
    data_label.append( 'ISTA' )
    pp.plot( 1 + np.arange(PLOT_ITERS+1),  F_ista_mean, label = 'ISTA', c = 'k', ls = ':' )
    data.append( F_ista_mean )
    pp.fill_between( 1 + np.arange(PLOT_ITERS+1), F_ista_mean - F_ista_std, F_ista_mean + F_ista_std, color = 'k', alpha = 0.3 )
    data_std.append( F_ista_std )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( f'FISTA $\\tau={fista_tau}$' )
    pp.plot( 1 + np.arange(PLOT_ITERS+1), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i % len(COLORS)] )
    data.append( Fhist_mean )
    pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
    data_std.append( Fhist_std )
##### LEARNED FISTA
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    if i in fista_ld_to_plot:
        Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
        Fhist -= Ffinal
        Fhist /= Ffinal
        Fhist_mean = np.mean( Fhist, axis = 0 )
        Fhist_std = np.std( Fhist, axis = 0 )
        data_label.append( labels_fista_ld[i] )
        pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = labels_fista_ld[i], c = COLORS[i % len(COLORS)] )
        data.append( Fhist_mean )
        pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
        data_std.append( Fhist_std )
##### DEEP OPTIMZATION
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy' )
    if not os.path.exists(res_filename):
        print(res_filename)
        raise ValueError()
    if i in deepopt_to_plot:
        Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
        Fhist -= Ffinal
        Fhist /= Ffinal
        Fhist_mean = np.mean( Fhist, axis = 0 )
        Fhist_std = np.std( Fhist, axis = 0 )
        data_label.append( labels_deepopt[i] )
        pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = labels_deepopt[i], c = COLORS[i % len(COLORS)], ls = '--' )
        data.append( Fhist_mean )
        pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
        data_std.append( Fhist_std )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'do_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'do_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'do_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
with open( os.path.join( FIG_PATH, 'do_optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")


######################### PLOTTING DEVIATION NORMS ##################################
pp.loglog( 1 + np.arange(PLOT_ITERS+1),  1e+3/(1 + np.arange(PLOT_ITERS+1)), label = '$O(1/k)$', c = 'k', ls = '-.' )
pp.loglog( 1 + np.arange(9,PLOT_ITERS+1),  1e+5/(1 + np.arange(9,PLOT_ITERS+1))**(1.5), label = '$O(1/k^{1.5})$', c = 'k', ls = ':' )
##### LEARNED FISTA
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'dyhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)] )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

    res_filename = os.path.join( RESULT_PATH[i], 'dwhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], label  = 'FISTA-LD ' + str(i+1) )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

##### DEEP OPTIMZATION
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx1hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], ls = '--' )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx2hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], ls = '--', label = 'DeepOpt ' + str(i+1) )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )
pp.xlabel('$k$')
pp.ylabel('mean$\\left( \\text{deviaton\'s norm} \\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'desvios_todos' + EXTENSION ) )
pp.close()


######################### PLOTTING UPPER BOUNDS TO DEVIATION NORMS ##################################
pp.loglog( 1 + np.arange(PLOT_ITERS+1),  1e+3/(1 + np.arange(PLOT_ITERS+1)), label = '$O(1/k)$', c = 'k', ls = '-.' )
pp.loglog( 1 + np.arange(9,PLOT_ITERS+1),  1e+5/(1 + np.arange(9,PLOT_ITERS+1))**(1.5), label = '$O(1/k^{1.5})$', c = 'k', ls = ':' )
##### LEARNED FISTA
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'dyubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist =  np.load( res_filename )[:,:PLOT_ITERS]

    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)] )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

    res_filename = os.path.join( RESULT_PATH[i], 'dwubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]

    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], label  = 'FISTA-LD ' + str(i+1) )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

##### DEEP OPTIMZATION
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx1ubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], ls = '--' )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx2ubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    hist = np.load( res_filename )[:,:PLOT_ITERS]
    hist_mean = np.mean( hist, axis = 0 )
    hist_std = np.std( hist, axis = 0 )
    pp.loglog( 1 + np.arange(hist.shape[1]), hist_mean, c = COLORS[i % len(COLORS)], ls = '--', label = 'DeepOpt ' + str(i+1) )
    pp.fill_between( 1 + np.arange(hist.shape[1]), hist_mean - hist_std, hist_mean + hist_std, color = COLORS[i % len(COLORS)], alpha = 0.3 )

pp.xlabel('$k$')
pp.ylabel('mean$\\left( \\text{deviaton\'s norm upper bound} \\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'limitante_desvios_todos' + EXTENSION ) )
pp.close()


######################### PLOTTING DEVIATION NORMS REL TO UPPER BOUNDS ##################################
##### LEARNED FISTA
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'dyhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    norm_hist = np.load( res_filename )[:,:PLOT_ITERS]
    res_filename = os.path.join( RESULT_PATH[i], 'dyubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    bound_hist = np.load( res_filename )[:,:PLOT_ITERS]
    pp.loglog( 1 + np.arange(norm_hist.shape[1]), np.mean( norm_hist / bound_hist, axis = 0 ), label = 'FISTA-LD ' + str(i+1), c = COLORS[i % len(COLORS)] )

    res_filename = os.path.join( RESULT_PATH[i], 'dwhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    norm_hist = np.load( res_filename )[:,:PLOT_ITERS]
    res_filename = os.path.join( RESULT_PATH[i], 'dwubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    bound_hist = np.load( res_filename )[:,:PLOT_ITERS]
    pp.loglog( 1 + np.arange(norm_hist.shape[1]), np.mean( norm_hist / bound_hist, axis = 0 ), c = COLORS[i % len(COLORS)] )

##### DEEP OPTIMZATION
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx1hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    norm_hist = np.load( res_filename )[:,:PLOT_ITERS]
    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx1ubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    bound_hist = np.load( res_filename )[:,:PLOT_ITERS]
    pp.loglog( 1 + np.arange(norm_hist.shape[1]), np.mean( norm_hist / bound_hist, axis = 0 ), label = 'DeepOpt ' + str(i+1), ls = '--', c = COLORS[i % len(COLORS)] )

    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx2hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    norm_hist = np.load( res_filename )[:,:PLOT_ITERS]
    res_filename = os.path.join( RESULT_PATH_DO[i], 'dx2ubhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    bound_hist = np.load( res_filename )[:,:PLOT_ITERS]
    pp.loglog( 1 + np.arange(norm_hist.shape[1]), np.mean( norm_hist / bound_hist, axis = 0 ), ls = '--', c = COLORS[i % len(COLORS)] )
pp.xlabel('$k$')
pp.ylabel('mean$\\left( \\frac{\\text{deviation\'s norm}}{\\text{upper bound}} \\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'desvios_rel_upperbound_todos' + EXTENSION ) )
pp.close()


######################### TABLE 1 ##################################
K = np.array( [10,20,30,40,50] )
# K = np.array( [100,250,500,1000] )

data_mean = []
data_std = []
data_label = []

# ISTA
if PLOT_ISTA:
    mean_ISTA = np.mean( F_ista[:,:PLOT_ITERS+1], axis = 0 )
    std_ISTA = np.std( F_ista[:,:PLOT_ITERS+1], axis = 0 )
    data_mean.append( mean_ISTA[K] )
    data_std.append( std_ISTA[K] )
    data_label.append( 'ISTA' )

# FISTA
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    mean_FISTA = np.mean( Ffista_all[i][:,:PLOT_ITERS+1], axis = 0 )
    std_FISTA = np.std( Ffista_all[i][:,:PLOT_ITERS+1], axis = 0 )
    data_mean.append( mean_FISTA[K] )
    data_std.append( std_FISTA[K] )
    data_label.append( f'FISTA tau={fista_tau}' )

# Trained algorithms -- FISTA-LD
for i in range( trained_fista_ld ):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    mean_test = np.mean( Fhist, axis = 0 )
    std_test = np.std( Fhist, axis = 0 )
    data_mean.append( mean_test[K] )
    data_std.append( std_test[K] )
    data_label.append( labels_fista_ld[i] )

##### Trained algorithms -- DeepOpt
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    mean_test = np.mean( Fhist, axis = 0 )
    std_test = np.std( Fhist, axis = 0 )
    data_mean.append( mean_test[K] )
    data_std.append( std_test[K] )
    data_label.append( labels_deepopt[i] )

data_mean = np.stack( data_mean )
data_std = np.stack( data_std )

use_bold = np.zeros_like(data_mean)
use_ul = np.zeros_like(data_mean)
for j in range(len(K)):
    i_min = np.argmin(data_mean[:,j])
    use_ul[:,j] = data_mean[:,j] <= data_mean[i_min,j]
    use_bold[:,j] = data_mean[:,j] - data_std[:,j] <= data_mean[i_min,j] + data_std[i_min,j]

print('\n----------------------------------------------------------------- Printing the table of mean optmality gap for each k - LaTeX format ----------------------------------------------------------------------')
line = ''
for k in K:
    line += f' & $k={k}$'
line += '\\\\'
print(line)
print('\\midrule')
for i in range(len(data_label)):
    line = data_label[i]
    for j in range(len(K)):
        x = data_mean[i,j]
        dx = data_std[i,j]
        if x == 0:
            power = 0.0
        else:
            power = - floor( log10(abs(x)) )
        if round(x*10**power, 1) >= 10:
            power -= 1
        if use_bold[i,j]:
            if use_ul[i,j]:
                line += f'  &  \\underline{{\\bf{{({x*10**power:.1f}$\\pm${dx*10**power:.1f})e{-power}}}}}'
            else:
                line += f'  &  \\bf{{({x*10**power:.1f}$\\pm${dx*10**power:.1f})e{-power}}}'
        else:
            line += f'  &  ({x*10**power:.1f}$\\pm${dx*10**power:.1f})e{-power}'
    line += '  \\\\'
    print(line)
print('\n---------------------------------------------------------------------------------------------------------------------------------------------------------------------')


######################### TABLE 2 - LOSS ##################################
max_iters = np.array( [20,50] )
data = np.zeros( ( trained_fista_ld + trained_deepopt, len(max_iters) ) )

# FISTA -- need to use here same tau used for training: tau=1.0 for exact proximal and tau=0.5 for inexact proximal
if PROBLEM == 'lstv':
    fista_tau_training = 0.5
else:
    fista_tau_training = 1.0
res_filename_FISTA = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau_training)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
if not os.path.exists(res_filename_FISTA):
    raise ValueError(res_filename_FISTA)

# Trained algorithms -- FISTA-LD
for i in range( trained_fista_ld ):
    for j,iters in enumerate(max_iters):
        res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
        if not os.path.exists(res_filename):
            raise ValueError()
        Fhist = np.load( res_filename )[:,1:iters+1]
        Ffista_hist = np.load( res_filename_FISTA )[:,1:iters+1]
        Ffista_final = np.load( res_filename_FISTA )[:,-1]
        Ffista_final = Ffista_final[:,None]
        data[i, j] = np.mean( ( Fhist - Ffista_final ) / ( Ffista_hist - Ffista_final ) )

##### Trained algorithms -- DeepOpt
for j,iters in enumerate(max_iters):
    for i in range( trained_deepopt ):
        res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy')
        if not os.path.exists(res_filename):
            raise ValueError()
        Fhist = np.load( res_filename )[:,1:iters+1]
        Ffista_hist = np.load( res_filename_FISTA )[:,1:iters+1]
        Ffista_final = np.load( res_filename_FISTA )[:,-1]
        Ffista_final = Ffista_final[:,None]
        data[trained_fista_ld+i, j] = np.mean( ( Fhist - Ffista_final ) / ( Ffista_hist - Ffista_final ) )

table2 = pd.DataFrame( data, index = labels_fista_ld + labels_deepopt, columns = [ f'[1,{iters}]' for iters in max_iters] )
table2.index.name = 'Algorithms'
table2.columns.name = 'Validation iterations'
print(table2.to_string( float_format="%.2e", col_space=12 ), '\n')
# print(table2.to_latex( float_format="%.2e" ), '\n')


######################### TABLE 3 - PENALTY ##################################
data = np.zeros( ( trained_fista_ld + trained_deepopt, ) )
maxiter = 20

# FISTA -- need to use here same tau used for training: tau=1.0 for exact proximal and tau=0.5 for inexact proximal
if PROBLEM == 'lstv':
    fista_tau_training = 0.5
else:
    fista_tau_training = 1.0
res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau_training)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
if os.path.exists(res_filename):
    Ffista_hist = np.load( res_filename )[:,maxiter+1:PLOT_ITERS+1]
else:
    raise ValueError(res_filename)

# Trained algorithms -- FISTA-LD
for i in range( trained_fista_ld ):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()

    Fhist = np.load( res_filename )[:,maxiter+1:PLOT_ITERS+1]

    penalty = np.mean( np.maximum( (Fhist - Ffista_hist)/Ffista_hist, 0.0 ) )
    data[i] = penalty

##### Trained algorithms -- DeepOpt
for i in range(trained_deepopt):
    res_filename = os.path.join( RESULT_PATH_DO[i], 'Fhist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError()

    Fhist = np.load( res_filename )[:,maxiter+1:PLOT_ITERS+1]

    penalty = np.mean( np.maximum( (Fhist - Ffista_hist)/Ffista_hist, 0.0 ) )
    data[trained_fista_ld+i] = penalty

table3 = pd.DataFrame( data, index = labels_fista_ld + labels_deepopt )
table3.index.name = 'Algorithms'
table3.columns.name = 'Penalty'
print(table3.to_string( float_format="%.2e", col_space=12 ), '\n')
# print(table3.to_latex( float_format="%.2e" ), '\n')
