import torch
import numpy as np

def mask( X, M, C, requires_grad=False ):
    if requires_grad:
        return ( X * ( 1 - M ) + C * M ).requires_grad_( True )
    else:
        return X * ( 1 - M ) + C * M

def mask_fun( X, M, C, x, y, classifier, loss_fn, 
        decoder = torch.nn.Identity(),
        prox_reg = 1.0, dpp_reg = 1.0, immutable_idxs=[] ):

    X_ = mask( X, M, C, requires_grad=True )
    result = loss( X_, M, C, x, y, classifier, loss_fn, 
                   prox_reg = 0., dpp_reg = dpp_reg )
    result.backward()

    X_.grad[ M.to( bool ) ] = 0
    X_.grad[ :, immutable_idxs ] = 0

    eps = abs( X_.grad ).argmax( axis=1 )
    M[ np.arange( M.shape[0] ), eps ] = 1
    return M

def loss( X_, M, C, x, y, classifier, loss_fn,
          prox_reg = 1.0, dpp_reg = 1.0, decoder=torch.nn.Identity() ):

    r = torch.sum( X_ * X_, axis=1 )
    r = torch.reshape(r, [-1,1])
    D = r - 2 * torch.matmul( X_, X_.T ) + r.T
    K = 1 / ( 1 + D )
    dpp = torch.det( K )

    prox = torch.norm( X_ - x, p=2, dim=1 )
    prox = prox.mean()

    l = loss_fn( classifier( decoder( X_ ) ), y )
    return l + prox_reg * prox - dpp_reg * dpp
    

def optimise_counterfactuals(reference, desired_output, classifier, loss_fn,
                    decoder=torch.nn.Identity(), 
                    num_counterfactuals=3, lr=1e-2, acc=0.95, iters=500,
                    prox_reg = 1.0, dpp_reg = 1.0, immutable_idxs=[],
                    non_decreasing_idxs=[], non_increasing_idxs=[] ):
    """

        Args:
            - reference: tuple of (sample_x, y) to use as reference for generating counterfactuals,
    """
 
    #reference output
    x = reference[0].flatten()
    y = desired_output # desired counterfactual output
    y_ = torch.tile( y, ( num_counterfactuals, 1 ) ).squeeze()

    C = 10 * torch.randn(num_counterfactuals, x.shape[0] )
    C.requires_grad_( True )
    l = []
    optim_C = torch.optim.Adam( (C,), lr=lr )

    M = torch.zeros( C.shape )
    X_ = mask( x, M, C )

    labels = classifier( decoder( X_ ) )

    while labels.mean( axis=0 )[ desired_output.to( bool ) ] < acc:
        M = mask_fun( x, M, C.detach(), x, y_, classifier,
                      loss_fn, decoder=decoder, 
                      immutable_idxs=immutable_idxs )

        for i in range( iters ):
            optim_C.zero_grad()
            
            X_ = mask( x, M, C )
            out = loss( X_, M, C, x, y_, classifier, loss_fn, decoder=decoder,
                    prox_reg = prox_reg, dpp_reg = dpp_reg )
            
            out.backward()

            optim_C.step()
            if i%100 == 0:
                print( "Iteration {}; Loss {}".format( i, out ) )

        labels = classifier( decoder( X_ ) )
        print( labels.mean( axis=0 ))
    X_ = mask( x, M, C )
    return decoder( X_ ).detach()
