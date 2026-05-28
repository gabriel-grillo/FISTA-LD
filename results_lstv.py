import os
import datetime
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors
from itertools import product

######################### PLOTTING OPTIONS ##################################
# pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
# pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\sen}{sen}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
COLORS = list(mcolors.TABLEAU_COLORS.keys())


######################### RESULTS PARAMETERS ##################################
MODE                = 'test'
DATASET             = 'mayo_clinic_512'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lstv'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso' or 'lstv'
EPOCHS_TO_PLOT      = [ 20 ]
DATE                = '2026-05-23'
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAINED_ITERS       = list(zip(MINITER,MAXITER))
PLOT_ITERS          = 200

TRAIN_DATASET_RATIO = [1.0]
TEST_DATASET_RATIO  = 1.0

### FISTA parameters
TAU_UNTRAINED_FISTA = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
# TAU_UNTRAINED_FISTA = [ 0.75 ]

### FISTA-LD parameters
ALPHA           = [ 10**(-2.5) ]
GAMMA           = [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
CONST_TAU       = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
FISTA_LD_PARAMS = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
# ALPHA           = [ 10**(-2.5) ]
# GAMMA           = [ 0.0 ]
# CONST_TAU       = [ 0.75 ]
# FISTA_LD_PARAMS = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
trained_fista_ld = len( list( product( FISTA_LD_PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_PLOT, TRAINED_ITERS ) ) )


######################### SETTING FIGURES' PATH ##################################
# SAVEDATE = '2026-05-25a'
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
Ffinal = np.min( np.stack( [ np.min(hist, axis=1) for hist in F_fistald_all ] ), axis = 0 )
##### FISTA RESULTS
Ffista_all = []
for fista_tau in TAU_UNTRAINED_FISTA:
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
    if os.path.exists(res_filename):
        Ffista_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
##### COMPARING FISTA AND FISTA-LD
Ffinal = np.minimum( Ffinal, np.min( np.stack( [ np.min(hist, axis=1) for hist in Ffista_all ] ), axis = 0 ) )
Ffinal = Ffinal[:,None]

######################### UNTRAINED FISTA AND ISTA RESULTS - RELATIVE OPT GAP ##################################
for i,_ in enumerate(TAU_UNTRAINED_FISTA):
    Ffista_all[i] -= Ffinal
    Ffista_all[i] /= Ffinal

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

PLOT_ITERS = min( Ffista_all[0].shape[1]-1, PLOT_ITERS )
######################### PLOTTING MEAN RELATIVE OPT GAP ##################################
data = [ 1 + np.arange(PLOT_ITERS+1) ]
data_std = [ 1 + np.arange(PLOT_ITERS+1) ]
data_label = [ 'k+1' ]
data_prox_failure_iter = []
pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    data_label.append( f'FISTA $\\tau={fista_tau}$' )
    pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
    data.append( Fhist_mean )
    # pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = COLORS[i], alpha = 0.5 )
    data_std.append( Fhist_std )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
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
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'flag_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    flag_hist = np.load( res_filename )[:,:PLOT_ITERS]
    if np.any( flag_hist == 0 ):
        pp.axvline( np.mean( np.argmin(flag_hist, axis = 1), where = np.any( flag_hist == 0, axis = 1) ),
                    c = COLORS[i], alpha = 0.5, label = f'prox failure ($\\tau={fista_tau}$)' )
        data_prox_failure_iter.append( np.round( np.mean( np.argmin(flag_hist, axis = 1), where = np.any( flag_hist == 0, axis = 1) ) ) )
    else:
        data_prox_failure_iter.append( -1 )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'flag_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    flag_hist = np.load( res_filename )[:,:PLOT_ITERS]
    if np.any( flag_hist == 0 ):
        pp.axvline( np.mean( np.argmin(flag_hist, axis = 1), where = np.any( flag_hist == 0, axis = 1) ),
                    c = COLORS[i%len(COLORS)], alpha = 0.5, label = labels_fista_ld[i] )
        data_prox_failure_iter.append( np.round( np.mean( np.argmin(flag_hist, axis = 1), where = np.any( flag_hist == 0, axis = 1) ) ) )
    else:
        data_prox_failure_iter.append( -1 )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'optgap' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
np.savetxt( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_prox_failure_iter.dat' ),
            np.column_stack(data_prox_failure_iter) )
with open( os.path.join( FIG_PATH, 'optgap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")

######################### PLOTTING MEAN R ##################################
data = [ 1 + np.arange(PLOT_ITERS+1) ]
data_std = [ 1 + np.arange(PLOT_ITERS+1) ]
data_label = [ 'k+1' ]
pp.plot( 1 + np.arange(PLOT_ITERS+1),  1e+6/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k' )
pp.plot( 1 + np.arange(9,PLOT_ITERS+1),  1e+9/(1 + np.arange(9,PLOT_ITERS+1))**3, label = '$O(1/k^3)$', c = 'k', ls = '--' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Rhist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError()
    Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Rhist_mean = np.mean( Rhist, axis = 0 )
    Rhist_std = np.std( Rhist, axis = 0 )
    data_label.append( f'FISTA $\\tau={fista_tau}$' )
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
pp.savefig( os.path.join( FIG_PATH, 'R_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'R_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data.dat' ),
            np.column_stack(data) )
np.savetxt( os.path.join( FIG_PATH, 'R_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_std.dat' ),
            np.column_stack(data_std) )
with open( os.path.join( FIG_PATH, 'R_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + 'data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")


######################### PLOTTING FLAG ##################################
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'flag_hist.npy' )
    if not  os.path.exists(res_filename):
        raise ValueError(res_filename)
    flag_hist = np.load( res_filename )[:,:PLOT_ITERS]
    pp.plot( 1 + np.arange(flag_hist.shape[1]), np.sum( flag_hist, axis = 0 ), label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'flag_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    flag_hist = np.load( res_filename )[:,:PLOT_ITERS+1]
    pp.plot( 1 + np.arange(flag_hist.shape[1]), np.sum( flag_hist, axis = 0 ), label = labels_fista_ld[i], c = COLORS[i%len(COLORS)] )
pp.xscale('log')
pp.xlabel('$k$')
pp.ylabel('sum$\\left(\\text{flag}_k\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'flag_count_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING MEAN INNER ITERS ##################################
pp.loglog( 1 + np.arange(10,99),  1e-2*(1 + np.arange(10,99))**2, label = '$O(k^2)$', c = 'k' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    inner_iter_hist_mean = np.mean( inner_iter_hist, axis = 0 )
    inner_iter_hist_std = np.std( inner_iter_hist, axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean - inner_iter_hist_std, inner_iter_hist_mean + inner_iter_hist_std, color = COLORS[i], alpha = 0.5 )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'inner_iter_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    inner_iter_hist_mean = np.mean( inner_iter_hist, axis = 0 )
    inner_iter_hist_std = np.std( inner_iter_hist, axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean, label = labels_fista_ld[i], c = COLORS[i%len(COLORS)] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean - inner_iter_hist_std, inner_iter_hist_mean + inner_iter_hist_std, color = COLORS[i%len(COLORS)], alpha = 0.5 )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k$')
pp.ylabel('mean$\\left(\\text{inner\\_iter}_k\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'inner_iter_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING MEAN CUMULATIVE INNER ITERS ##################################
pp.loglog( 1 + np.arange(10,200),  1e-2*(1 + np.arange(10,200))**3, label = '$O(k^3)$', c = 'k' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]    
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    cum_inner_iter_hist_std = np.std( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean - cum_inner_iter_hist_std, cum_inner_iter_hist_mean + cum_inner_iter_hist_std, color = COLORS[i], alpha = 0.5 )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'inner_iter_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    cum_inner_iter_hist_std = np.std( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean, label = labels_fista_ld[i], c = COLORS[i%len(COLORS)] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean - cum_inner_iter_hist_std, cum_inner_iter_hist_mean + cum_inner_iter_hist_std, color = COLORS[i%len(COLORS)], alpha = 0.5 )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k$')
pp.ylabel('mean$\\left(\\text{cum\\_inner\\_iter}_k\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'cum_inner_iter_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING MEAN CUMULATIVE INNER ITERS X MEAN OPT GAP ##################################
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]    
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( cum_inner_iter_hist_mean, Fhist_mean[1:], label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    Fhist_mean = np.mean( Fhist, axis = 0 )
    res_filename = os.path.join( RESULT_PATH[i], 'inner_iter_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( cum_inner_iter_hist_mean, Fhist_mean[1:], label = labels_fista_ld[i], c = COLORS[i%len(COLORS)] )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('mean$\\left(\\text{cum\\_inner\\_iter}_k\\right)$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'cum_inner_iter_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING MEAN COST ESTIMATE X MEAN OPT GAP ##################################
### UNITIES OF COST
u_o_fista = 9.9e-3
u_o_fistald = 1.6e-2
u_i = 1.8e-3
data_cost = []
data_gap = []
data_label = []
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]    
    cost_hist =  u_o_fista * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    cost_hist_mean = np.mean( cost_hist, axis = 0 )
    cost_hist_std = np.std( cost_hist, axis = 0 )
    y_error = [ Fhist_mean[1:] - Fhist_std[1:], Fhist_mean[1:] + Fhist_std[1:] ]
    x_error = [ cost_hist_mean - cost_hist_std, cost_hist_mean + cost_hist_std ]
    pp.plot( cost_hist_mean, Fhist_mean[1:], label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = COLORS[i] )
    data_cost.append( cost_hist_mean )
    data_gap.append( Fhist_mean[1:] )
    data_label.append( f'FISTA $\\tau={fista_tau}$' )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    res_filename = os.path.join( RESULT_PATH[i], 'inner_iter_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    cost_hist =  u_o_fistald * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    cost_hist_mean = np.mean( cost_hist, axis = 0 )
    cost_hist_std = np.std( cost_hist, axis = 0 )
    y_error = [ Fhist_mean[1:] - Fhist_std[1:], Fhist_mean[1:] + Fhist_std[1:] ]
    x_error = [ cost_hist_mean - cost_hist_std, cost_hist_mean + cost_hist_std ]
    pp.plot( cost_hist_mean, Fhist_mean[1:], label = labels_fista_ld[i], c = COLORS[i%len(COLORS)] )
    data_cost.append( cost_hist_mean )
    data_gap.append( Fhist_mean[1:] )
    data_label.append( labels_fista_ld[i] )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('mean(cost_k)')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'cost_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'cost_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + '_cost.dat' ),
            np.column_stack(data_cost) )
np.savetxt( os.path.join( FIG_PATH, 'cost_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + '_gap.dat' ),
            np.column_stack(data_gap) )
with open( os.path.join( FIG_PATH, 'cost_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + '_data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")

######################### PERFORMANCE PROFILE ##################################
rel_error_thr = 1e-1
problems = Ffista_all[0].shape[0]
perf_list = []
perf_prof_label = []

for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,1:]
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if not os.path.exists(res_filename):
        raise ValueError(res_filename)
    inner_iter_hist = np.load( res_filename )
    cost_hist =  u_o_fista * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    perf = np.inf * np.ones( (problems,) )
    for j in range( problems ):
        if np.any( Fhist[j] < rel_error_thr ):
            thr_iter = np.argmax( Fhist[j] < rel_error_thr )
            perf[j] = cost_hist[j,thr_iter]
    perf_list.append(perf)
    perf_prof_label.append( f'FISTA $\\tau={fista_tau}$' )
for i in range(trained_fista_ld):
    res_filename = os.path.join( RESULT_PATH[i], 'Fhist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    Fhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    Fhist -= Ffinal
    Fhist /= Ffinal
    res_filename = os.path.join( RESULT_PATH[i], 'inner_iter_hist.npy')
    if not os.path.exists(res_filename):
        raise ValueError()
    inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    cost_hist =  u_o_fistald * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    perf = np.inf * np.ones( (problems,) )
    for j in range( problems ):
        if np.any( Fhist[j] < rel_error_thr ):
            thr_iter = np.argmax( Fhist[j] < rel_error_thr )
            perf[j] = cost_hist[j,thr_iter]
    perf_list.append(perf)
    perf_prof_label.append( labels_fista_ld[i] )

perf_mat = np.stack( perf_list )
algorithms = len(perf_list)

mincost_perf_mat = np.min( perf_mat, axis = 0 )
maxcost_perf_mat = np.max( perf_mat, axis = 0, where=perf_mat!=np.inf, initial=-1.0 )
kappas = np.linspace( 1, 1.01 * np.max( maxcost_perf_mat / mincost_perf_mat ), num = 500 ) # multiply by 1.01 to avoid missing the maxcost in the plot by numerical reasons

fractions_to_plot = np.zeros( (algorithms, kappas.size) )
for i in range(algorithms):
    for l,kappa in enumerate(kappas):
        fractions_to_plot[i,l] = np.sum( perf_mat[i] <= kappa * mincost_perf_mat ) / problems

for i, label in enumerate( perf_prof_label ):
    pp.plot( kappas, fractions_to_plot[i], label = label )
pp.title( f'Performance profile with tolerance {rel_error_thr:.2e}' )
pp.xlabel('Performance ratio (estimated cost)')
pp.ylabel('Fraction of problems')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.xlim(left=1)
pp.ylim(bottom=0.0)
pp.savefig( os.path.join( FIG_PATH, 'perf_prof' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()
np.savetxt( os.path.join( FIG_PATH, 'perf_prof_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + '_data.dat' ),
            np.column_stack([kappas, fractions_to_plot.T]) )
data_label = ['kappas'] + perf_prof_label
with open( os.path.join( FIG_PATH, 'perf_prof_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int( TEST_DATASET_RATIO * 100 ) ) + 'percent' + '_data_label.txt' ), 'w' ) as f:
    for line in data_label:
        f.write(f"{line}\n")