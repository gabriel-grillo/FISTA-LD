import tensorflow as tf
from proxTV import proxTV, reduce_TV

#############################
# Optimize F(x) = f(x) + g(x)
#############################

##### Helpers

def dot(x,y):
    dot = tf.reduce_sum( x * y, axis=[1,2,3] )
    return dot
def norm_sq(x):
    norm = tf.reduce_sum( x**2, axis = [1,2,3] )
    return norm
def safe_sqrt(x):
    safe_sqrt = tf.sqrt(tf.where(tf.equal(x, 0.0), x + 1, x))
    return tf.where(tf.equal(x, 0.0), tf.zeros_like(x), safe_sqrt)
## Safe normalization ensures that there is no division by 0 in backpropagation
def safely_normalize( x, M ):
    sqnorm = tf.reduce_sum( x**2, axis=[1,2,3], keepdims=True )
    safe_sqrt = tf.sqrt(sqnorm + M ** 2)
    x = M * x / safe_sqrt
    return x

##### Objective function terms
def huber(inp, gamma):
    batch_size = tf.shape(inp)[0]
    y = tf.where(tf.abs(inp) < gamma, 
                 0.5 * inp**2 / gamma, 
                 tf.abs(inp) - 0.5 * gamma)
    return tf.reduce_sum( tf.reshape( y, shape = [batch_size, -1] ), axis = -1 )
def huber_grad(inp, gamma):
    return tf.where(tf.abs(inp) < gamma, inp / gamma, tf.sign(inp))

def l1( inp ):
    out = tf.reduce_sum( tf.abs( inp ), axis = [1,2,3] )
    return out
def prox_l1(x, alpha):
    res = tf.sign(x) * tf.maximum(tf.abs(x) - alpha, 0)
    return res

def objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam ):
    if problem == 'nnls':
        # Nonnegative least squares problem
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return 0.0 # I always project anyway
        def prox( inp, step_size, eps ):
            out = tf.math.maximum( 0.0, inp )
            return out, tf.fill( [tf.shape(inp)[0]], 1 ), tf.fill( [tf.shape(inp)[0]], 1 )
    elif problem == 'lasso':
        # Lasso problem: W-l1 regularized least squares problem
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return l1( lam * W(x) )
        def prox(x, rho, eps):
            return W_adj( prox_l1( W(x), lam * rho ) ), tf.fill( [tf.shape(x)[0]], 1 ), tf.fill( [tf.shape(x)[0]], 1 )
    elif problem == 'slasso':
        # Smoothed Lasso problem: W-l1 (smoothed) regularized least squares problem
        huber_param = tf.constant( 1e-2, dtype = tf.float32, name = 'huber_param' )
        def f( x ):
            r = A( x ) - b
            Wx = W( x )
            def grad():
                return A_adj( r ) + lam * W_adj( huber_grad( Wx, huber_param ) )
            return 0.5 * norm_sq( r ) + lam * huber( Wx, huber_param ), grad
        L = A_norm**2 + lam * (W_norm**2) / huber_param
        def g( x ):
            return 0.0
        def prox( x, rho, eps):
            return x, tf.fill( [tf.shape(x)[0]], 1 ), tf.fill( [tf.shape(x)[0]], 1 )
    elif problem == 'nnslasso':
        # Nonnegative smoothed Lasso problem: W-l1 (smoothed) regularized least squares problem
        huber_param = tf.constant( 1e-2, dtype = tf.float32, name = 'huber_param' )
        def f( x ):
            r = A( x ) - b
            Wx = W(x)
            def grad():
                return A_adj( r ) + lam * W_adj( huber_grad( Wx, huber_param ) )
            return 0.5*norm_sq( r ) + lam * huber( Wx, huber_param ), grad
        L = A_norm**2 + lam * (W_norm**2) / huber_param
        def g( x ):
            return 0.0 # I always project anyway
        def prox( inp, step_size, eps ):
            out = tf.math.maximum( 0.0, inp )
            return out, tf.fill( [tf.shape(inp)[0]], 1 ), tf.fill( [tf.shape(inp)[0]], 1 )
    elif problem == 'lstv':
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return lam * reduce_TV( W( x ) )
        prox = proxTV( W, W_adj, W_norm, lam )
    else:
        raise ValueError('Unimplemented problem')

    return f, g, prox, L


################# Computing R0 for fista_ld
def computeR( problem, dataset, x_shape, A, A_adj, A_norm, W, W_adj, W_norm, lam ):
    ## Running over dataset to performe one proximal gradient step
    R = tf.constant( 0.0, dtype = tf.float32 )
    for (b,_) in dataset:
        x0 = tf.zeros( (b.shape[0],) + x_shape, dtype = tf.float32 )
        x1, _, _ = ista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                         tf.constant( 1, dtype=tf.int32 ), save_hist = False
                       )
        R += tf.reduce_mean( norm_sq( x1 - x0 ) )
    R /= tf.cast( dataset.cardinality(), tf.float32 )
    ## Smoothness parameter
    _, _, _, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    R *= L

    return R


########################## FISTA with Learned Deviations ##########################
def fista_ld( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
              dy_model, dw_model,
              MAXITER, alpha, R0,
              var_tau = True, gamma = 0.05, const_tau = 0.5,
              training = False, save_hist = True
             ):
    
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    ## Objective funtion terms
    f, g, prox, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    
    ## Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    y = x
    t = tf.constant(1.0, dtype=tf.float32, name = 't')
    dy = tf.zeros_like(y)
    dw = tf.zeros_like(y)
    flag = tf.fill( [batch_size], 0, name = 'flag' )
    fx, gradx = f( x )
    grad = gradx()

    ## Parameter initializations
    if problem == 'lstv':
        beta_3 = 0.5
    else:
        beta_3 = 0.0
    if var_tau:
        tau = 0.99 / ( 1.0 + tf.math.exp( -gamma * 0.0 ) )
    else:
        tau = const_tau
    # Scale adjustment of R0
    R0 /= 2 * ( tau / L ) * (1-tau)
    R = tf.fill( [batch_size], R0 )

    if save_hist:
        Fx = fx + g( x )
    else:
        Fx = fx # shape initialization only
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        Rhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Rhist'           )
        dy_hist         = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dy_hist'         )
        dy_ub_hist      = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dy_ub_hist'      )
        dw_hist         = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dw_hist'         )
        dw_ub_hist      = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dw_ub_hist'      )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
        Rhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( R, axis = 1 )
    else:
        Rhist           = []
        Fhist           = []
        dy_hist         = []
        dy_ub_hist      = []
        dw_hist         = []
        dw_ub_hist      = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Saves previous iteration
        x_prev = x

        ## Deviations

        beta_1 = tau * ( 1 - beta_3 )
        beta_2 = ( 1 - tau ) * ( 1 - beta_3 )

        # dy and y tilde
        dy = dy_model( [y, grad, dy], training = training )
        dy_norm_upper_bound = safe_sqrt( 2.0 * beta_1 * R / ( L * ( 1.0 - tau ) ) )
        dy = safely_normalize( dy, dy_norm_upper_bound[:,tf.newaxis,tf.newaxis,tf.newaxis] )
        y_tilde = y + (1-tau) * dy
        # Current gradient
        _, grady_tilde = f( y_tilde )
        grad = grady_tilde()
        # dw
        dw = dw_model( [y, grad, dy, dw], training = training )
        dw_norm_upper_bound = safe_sqrt( 2.0 * tau * beta_2 * R / L )
        dw = safely_normalize( dw, dw_norm_upper_bound[:,tf.newaxis,tf.newaxis,tf.newaxis] )

        ## Performs proximal gradient step

        # Gradient step
        w = y - (tau/L) * grad
        w += dw
        
        # Proximal step
        x, flag, inner_iter = prox( w, tau/L, beta_3 * R )

        ## Warming up to the next iteration

        # Computes tau for next iteration
        if var_tau:
            tau_next = 0.99 / ( 1.0 + tf.math.exp( -gamma * ( tf.cast(i+1,tf.float32) ) ) )
        else:
            tau_next = tau

        # Computes t for next iteration
        t_next = (1.0 + tf.sqrt(1.0 + 4.0 * (tau/tau_next) * t**2)) / 2.0
        
        # Computes tolerance for next iteration
        R = ( ( 1.0 - tau ) / ( 2.0 * tau ) ) * L * norm_sq( x - y + tau * dy )
        R *= ( 1.0 - alpha ) * ( tau / tau_next ) * ( (t/t_next) ** 2 )
        
        # Computes y for next iteration
        y = x + ((t-1)/t_next) * (x - x_prev)
        y -= (t/t_next) * dw

        # Updates i, t, and tau for next iteration
        i += 1
        t = t_next
        tau = tau_next
        
        ## Saves history
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist      += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            Rhist      += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( R, axis = 1 )
            dy_hist    += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( ( 1.0 - tau ) * tf.sqrt( tf.reduce_sum( dy**2, axis = [1,2,3] ) ), axis = 1 )
            dy_ub_hist += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( ( 1.0 - tau ) * dy_norm_upper_bound, axis = 1 )
            dw_hist    += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( dw**2, axis = [1,2,3] ) ), axis = 1 )
            dw_ub_hist += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( dw_norm_upper_bound, axis = 1 )
            flag_hist  += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)
    
    return x, Fx, [Fhist, Rhist, dy_hist, dy_ub_hist, dw_hist, dw_ub_hist, flag_hist, inner_iter_hist]
    

########################## Baseline algorithms ##########################
def fista( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
           MAXITER, alpha, tau, R0, save_hist = True
          ):
    
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    ## Objective funtion terms
    f, g, prox, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    
    ## Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    y = x
    t = tf.constant(1.0, dtype=tf.float32, name = 't')
    flag = tf.fill( [batch_size], 0, name = 'flag' )

    fx, gradx = f( x )
    grad = gradx()

    ## Scale adjustment of R0
    R0 /= 2 * ( tau / L ) * (1-tau)
    R = tf.fill( [batch_size], R0 )

    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        Rhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Rhist'           )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        Fx = fx + g( x )
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
        Rhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( R, axis = 1 )
    else:
        Rhist     = []
        Fhist     = []
        flag_hist = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Saves previous iteration
        x_prev = x

        ## Performs proximal gradient step

        # Current gradient
        _, grady = f( y )
        grad = grady()

        # Gradient step
        w = y - (tau/L) * grad
        
        # Proximal step
        x, flag, inner_iter = prox( w, tau/L, R )

        ## Warming up to the next iteration
        
        # Computes t for next iteration
        t_next = (1.0 + tf.sqrt(1.0 + 4.0 * t**2)) / 2.0
        
        # Computes tolerance for next iteration
        R = ( ( 1.0 - tau ) / ( 2.0 * tau ) ) * L * norm_sq( x - y )
        R *= ( 1.0 - alpha ) *  ( (t/t_next) ** 2 )
        
        # Computes y for next iteration
        y = x + ((t-1)/t_next) * (x - x_prev)

        # Updates t and i for next iteration
        i += 1
        t = t_next

        ## Saves hist
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            Rhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( R, axis = 1 )
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)

    return x, Fx, [Fhist, Rhist, flag_hist, inner_iter_hist]

def ista( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
          MAXITER, save_hist = True
         ):
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    # Objective funtion terms
    f, g, prox, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )

    # Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        Rhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Rhist'           )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )
    else:
        Fhist           = []
        Rhist           = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Performs proximal gradient step

        # Current gradient
        fx, gradx = f( x )
        grad = gradx()

        # Saves Fx
        if save_hist:
            Fx = fx + g(x)
            Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )

        # Proximal gradient step
        x, flag, inner_iter = prox( x - (1/L) * grad, 1/L, tf.constant( 0.0, dtype = tf.float32 ) )

        ## Updates i for next iteration
        i += 1

        ## Saves hist
        if save_hist:
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)

    # Saves Fx
    if save_hist:
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    
    return x, Fx, [Fhist, Rhist, flag_hist, inner_iter_hist]


########################## Deep Optimization ##########################
##### based on implementations at https://github.com/JevgenijaAksjonova/Deep-Optimization
def deepopt_nonsmooth( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                       dx1_model, dx2_model, MAXITER, alpha,
                       training = False, save_hist = True ):
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    # Objective funtion terms
    f, g, prox, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )

    # Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    x_prev = x
    fx, grad_w = f(x)
    grad = grad_w()
    dx1 = tf.zeros_like(x)
    dx2 = tf.zeros_like(x)
    beta = 1/L    # Inverse of the smoothness parameter
    gamma = beta  # step-size
    omega = alpha # alpha_2
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        dx1_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx1_hist'        )
        dx1_ub_hist     = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx1_ub_hist'     )
        dx2_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx2_hist'        )
        dx2_ub_hist     = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx2_ub_hist'     )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        Fx = fx + g(x)
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    else:
        Fhist           = []
        dx1_hist        = []
        dx1_ub_hist     = []
        dx2_hist        = []
        dx2_ub_hist     = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Deviations
        dx1_prev = dx1
        dx2_prev = dx2
        # Compute dx1 and w
        dx1 = dx1_model( [x, grad, dx1], training = training )
        c1 = tf.sqrt(2 * beta * alpha * (2 * beta - gamma) / (2 * beta * gamma))
        c1 *=  safe_sqrt(norm_sq( x - x_prev - (beta / (2 * beta - gamma)) * dx2_prev ))
        dx1 = safely_normalize( dx1, c1[:,tf.newaxis,tf.newaxis,tf.newaxis]  )
        w = x + dx1
        # Compute current gradient
        grad_prev = grad
        _, grad_w = f(w)
        grad = grad_w()
        # Compute dx2
        dx2 = dx2_model( [x, grad, dx2, dx1], training = training )
        c2 = tf.sqrt(2 * gamma * (2 * beta - gamma) / beta * beta * omega / 2)
        c2 *= safe_sqrt(norm_sq( grad - grad_prev - (1 / beta) * (x - x_prev - dx1_prev) ))
        dx2 = safely_normalize( dx2, c2[:,tf.newaxis,tf.newaxis,tf.newaxis] )

        ## Performs proximal gradient step
        x_prev = x
        x -= gamma * grad
        x += (gamma / beta) * dx1
        x += dx2
        x, flag, inner_iter = prox( x, gamma, tf.constant( 0.0, dtype = tf.float32 ) )

        ## Updates i for next iteration
        i += 1

        # Saves Fx
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist       += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            dx1_hist    += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( dx1**2, axis = [1,2,3] ) ), axis = 1 )
            dx1_ub_hist += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( c1, axis = 1 )
            dx2_hist    += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( dx2**2, axis = [1,2,3] ) ), axis = 1 )
            dx2_ub_hist += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( c2, axis = 1 )
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )
    
    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)
    
    return x, Fx, [Fhist, dx1_hist, dx1_ub_hist, dx2_hist, dx2_ub_hist, flag_hist, inner_iter_hist ]
