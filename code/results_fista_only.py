import os
import datetime
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors

######################### PLOTTING OPTIONS ##################################
# pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
# pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\sen}{sen}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
FISTA_COLORS = list(mcolors.TABLEAU_COLORS.keys())

######################### RESULTS PARAMETERS ##################################
MODE       = 'val'
DATASET    = 'mayo_clinic_512'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM    = 'lstv'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso' or 'lstv'
PLOT_ITERS = 200

TEST_DATASET_RATIO  = 1.0

TAU_UNTRAINED_FISTA = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]

######################### SETTING FIGURES' PATH ##################################
# SAVEDATE = '2026-05-25b'
SAVEDATE = datetime.today().strftime('%Y-%m-%d')
FIG_PATH = os.path.join(os.getcwd(), 'plots', DATASET, PROBLEM, SAVEDATE )
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)

######################### COMPUTING F* ##################################
##### FISTA RESULTS
Ffista_all = []
for fista_tau in TAU_UNTRAINED_FISTA:
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Fhist.npy' )
    if os.path.exists(res_filename):
        Ffista_all.append( np.load( res_filename ) )
    else:
        raise ValueError(res_filename)
Ffista_all = np.stack( Ffista_all )
Ffinal = np.min( np.min(Ffista_all, axis = 2), axis = 0 )
Ffinal = Ffinal[:,None]

######################### UNTRAINED FISTA AND ISTA RESULTS - RELATIVE OPT GAP ##################################
for i,_ in enumerate(TAU_UNTRAINED_FISTA):
    Ffista_all[i] -= Ffinal
    Ffista_all[i] /= Ffinal

PLOT_ITERS = min( Ffista_all[0].shape[1]-1, PLOT_ITERS )
######################### PLOTTING MEAN RELATIVE OPT GAP ##################################
pp.plot( 1 + np.arange(PLOT_ITERS+1),  6000/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k', ls = '--' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    pp.plot( 1 + np.arange(Fhist.shape[1]), Fhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
    pp.fill_between( 1 + np.arange(Fhist.shape[1]), Fhist_mean - Fhist_std, Fhist_mean + Fhist_std, color = FISTA_COLORS[i%len(FISTA_COLORS)], alpha = 0.5 )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'flag_hist.npy' )
    if os.path.exists(res_filename):
        flag_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    pp.axvline( np.mean( np.where( np.any( flag_hist == 0, axis = 1) , np.argmin(flag_hist, axis = 1), PLOT_ITERS ) ),
                c = FISTA_COLORS[i%len(FISTA_COLORS)], alpha = 0.5, label = f'prox failure ($\\tau={fista_tau}$)' )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'optgap' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING MEAN R ##################################
pp.loglog( 1 + np.arange(PLOT_ITERS+1),  1e+6/(1 + np.arange(PLOT_ITERS+1))**2, label = '$O(1/k^2)$', c = 'k' )
pp.loglog( 1 + np.arange(9,PLOT_ITERS+1),  1e+9/(1 + np.arange(9,PLOT_ITERS+1))**3, label = '$O(1/k^3)$', c = 'k', ls = '--' )
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'Rhist.npy' )
    if os.path.exists(res_filename):
        Rhist = np.load( res_filename )[:,:PLOT_ITERS+1]
    else:
        raise ValueError(res_filename)
    Rhist_mean = np.mean( Rhist, axis = 0 )
    Rhist_std = np.std( Rhist, axis = 0 )
    pp.plot( 1 + np.arange(Rhist.shape[1]), Rhist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
    pp.fill_between( 1 + np.arange(Rhist.shape[1]), Rhist_mean - Rhist_std, Rhist_mean + Rhist_std, color = FISTA_COLORS[i%len(FISTA_COLORS)], alpha = 0.5 )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('$k+1$')
pp.ylabel('mean$\\left(R_k\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'R_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PLOTTING FLAG ##################################
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'flag_hist.npy' )
    if os.path.exists(res_filename):
        flag_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    pp.plot( 1 + np.arange(flag_hist.shape[1]), np.sum( flag_hist, axis = 0 ), label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
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
    if os.path.exists(res_filename):
        inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    inner_iter_hist_mean = np.mean( inner_iter_hist, axis = 0 )
    inner_iter_hist_std = np.std( inner_iter_hist, axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), inner_iter_hist_mean - inner_iter_hist_std, inner_iter_hist_mean + inner_iter_hist_std, color = FISTA_COLORS[i%len(FISTA_COLORS)], alpha = 0.5 )
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
    if os.path.exists(res_filename):
        inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    cum_inner_iter_hist_std = np.std( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean, label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
    pp.fill_between( 1 + np.arange(inner_iter_hist.shape[1]), cum_inner_iter_hist_mean - cum_inner_iter_hist_std, cum_inner_iter_hist_mean + cum_inner_iter_hist_std, color = FISTA_COLORS[i%len(FISTA_COLORS)], alpha = 0.5 )
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
    Fhist_std = np.std( Fhist, axis = 0 )
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if os.path.exists(res_filename):
        inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    cum_inner_iter_hist_mean = np.mean( np.cumsum(inner_iter_hist, axis = 1), axis = 0 )
    pp.plot( cum_inner_iter_hist_mean, Fhist_mean[1:], label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
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
u_o = 9.9e-3
u_i = 1.8e-3
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,:PLOT_ITERS+1]
    Fhist_mean = np.mean( Fhist, axis = 0 )
    Fhist_std = np.std( Fhist, axis = 0 )
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if os.path.exists(res_filename):
        inner_iter_hist = np.load( res_filename )[:,:PLOT_ITERS]
    else:
        raise ValueError(res_filename)
    cost_hist =  u_o * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    cost_hist_mean = np.mean( cost_hist, axis = 0 )
    cost_hist_std = np.std( cost_hist, axis = 0 )
    y_error = [ Fhist_mean[1:] - Fhist_std[1:], Fhist_mean[1:] + Fhist_std[1:] ]
    x_error = [ cost_hist_mean - cost_hist_std, cost_hist_mean + cost_hist_std ]
    pp.plot( cost_hist_mean, Fhist_mean[1:], label = f'FISTA $\\tau={fista_tau}$', ls = '-.', c = FISTA_COLORS[i%len(FISTA_COLORS)] )
pp.xscale('log')
pp.yscale('log')
pp.xlabel('mean(cost_k)')
pp.ylabel('mean$\\left(\\frac{F(x_k)-F^*}{F^*}\\right)$')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.savefig( os.path.join( FIG_PATH, 'cost_x_gap_' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()

######################### PERFORMANCE PROFILE ##################################
problems = Ffista_all.shape[1]
algorithms = Ffista_all.shape[0]
perf_mat = np.zeros( ( algorithms, problems ) )
rel_error_thr = 1e-1

for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    Fhist = Ffista_all[i][:,1:]
    res_filename = os.path.join( 'test_results', MODE, 'untrained', 'fista', 'tau' + str(fista_tau)[2:], DATASET, str( TEST_DATASET_RATIO * 100 ) + 'percent', PROBLEM, 'inner_iter_hist.npy' )
    if os.path.exists(res_filename):
        inner_iter_hist = np.load( res_filename )
    else:
        raise ValueError(res_filename)
    cost_hist =  u_o * (1 + np.arange(inner_iter_hist.shape[1])) + u_i * np.cumsum( inner_iter_hist, axis = 1)
    for j in range( problems ):
        if np.any( Fhist[j] < rel_error_thr ):
            thr_iter = np.argmax( Fhist[j] < rel_error_thr )
            perf_mat[i,j] = cost_hist[j,thr_iter]
        else:
            perf_mat[i,j] = np.inf

mincost_perf_mat = np.min( perf_mat, axis = 0 )
maxcost_perf_mat = np.max( perf_mat, axis = 0, where=perf_mat!=np.inf, initial=-1.0 )
kappas = np.linspace( 1, 1.01 * np.max( maxcost_perf_mat / mincost_perf_mat ), num = 500 ) # multiply by 1.01 to avoid missing the maxcost in the plot by numerical reasons

fractions_to_plot = np.zeros( (algorithms, kappas.size) )
for i in range(algorithms):
    for l,kappa in enumerate(kappas):
        fractions_to_plot[i,l] = np.sum( perf_mat[i] <= kappa * mincost_perf_mat ) / problems
    
for i,fista_tau in enumerate(TAU_UNTRAINED_FISTA):
    pp.plot( kappas, fractions_to_plot[i], label = f'FISTA $\\tau={fista_tau}$', ls = '-.',  c = FISTA_COLORS[i%len(FISTA_COLORS)] )
pp.title( f'Performance profile with tolerance {rel_error_thr:.2e}' )
pp.xlabel('Performance ratio (estimated cost)')
pp.ylabel('Fraction of problems')
pp.grid()
pp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pp.xlim(left=1)
pp.ylim(bottom=0.0)
pp.savefig( os.path.join( FIG_PATH, 'perf_prof' + PROBLEM + '_' + DATASET + '_' + MODE + '_' + str( int(TEST_DATASET_RATIO * 100) ) + 'percent' + EXTENSION ) )
pp.close()
