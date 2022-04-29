import torch
import numpy as np

#Mothital et al, loss
def loss( C, x, y, classifier, loss_fn, mad=1, cov=None,
          prox_reg = 1.0, dpp_reg = 1.0, decoder=torch.nn.Identity(),
          gaussian_reg = None ):

    if cov is None:
        cov = torch.eye( x.shape[0] )

    r = torch.sum( C * C, axis=1 )

    r = torch.reshape(r, [-1,1])
    D = r - 2 * torch.matmul( C, C.T ) + r.T
    K = 1 / ( 1 + D )
    dpp = torch.det( K )

    L = torch.linalg.cholesky( torch.inverse( cov ) )
    prox = torch.norm( C @ L - x, p=2, dim=0 )**2
    prox = prox.mean()

    if gaussian_reg is not None:
        g_reg = - gaussian_reg.log_prob( C )
        g_reg = g_reg.mean()
    else:
        g_reg = 0
    l = loss_fn( classifier(C ).reshape(-1, 2), y.reshape(-1, 2) )
    norm = 1#torch.norm( torch.tensor( [ l, prox, dpp ] ) )
    return ( l + prox_reg * prox + dpp_reg * dpp + g_reg ) / norm
    

def optimise_counterfactuals(reference, desired_output, classifier, loss_fn,
                    decoder=torch.nn.Identity(), 
                    num_counterfactuals=3, lr=1e-2, mad=1, cov = None,
                    gaussian_reg = None,
                    iters=500, prox_reg = 1.0, dpp_reg = 1.0, 
                    immutable_idxs=[], non_decreasing_idxs=[], 
                    non_increasing_idxs=[] ):
    """

        Args:
            - reference: tuple of (sample_x, y) to use as reference for generating counterfactuals,
    """
    if cov is None:
        cov = torch.eye( reference[0].shape[0] )

    #reference output
    x = reference[0].flatten()

    y = desired_output # desired counterfactual output
    y_ = torch.tile( y, ( num_counterfactuals, 1 ) ).squeeze()

    C = torch.rand(num_counterfactuals, x.shape[0] )
    C[ :, immutable_idxs ] = x[ immutable_idxs ]
    C.requires_grad_( True )
    
    l = []
    optim_C = torch.optim.Adam( (C,), lr=lr )
    for i in range( iters ):
        optim_C.zero_grad()

        out = loss( C, x, y_, classifier, loss_fn, cov=cov, decoder=decoder,
                prox_reg = prox_reg, dpp_reg = dpp_reg, gaussian_reg = gaussian_reg )
        
        out.backward()
        C.grad[ :, immutable_idxs ] = 0.
        
        optim_C.step()
        if i%100 == 0:
            print( "Iteration {}; Loss {}".format( i, out ) )

    return C.detach()
