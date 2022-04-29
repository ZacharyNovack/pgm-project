import torch
from torch.autograd.functional import hessian

from gaussian import Gaussian

from sklearn.linear_model import LogisticRegression as log_reg
import numpy as np

class CounterfactualDistribution( Gaussian ):
    def __init__( self, mu, cov, alpha=0.5 ):
        super().__init__( mu, cov )
        
        self.set_rep()
        self.set_dec()
        self.set_enc()

        x_rep = self.rep( self.dec( self.sample([1]) ) )
        self.set_bayesian_weights( )

    def set_rep( self, rep = torch.nn.Identity() ):
        self.rep = rep
        
    def set_dec( self, dec = torch.nn.Identity() ):
        self.dec = dec
    
    def set_enc( self, enc = torch.nn.Identity() ):
        self.enc = enc
    
    def set_data( self, known_data, known_labels ):
        self.known_data = torch.hstack( [ 
                self.rep( known_data ),
                torch.ones( known_data.shape[0], 1 ) ] )

        if known_labels.shape[1] <= 1:
            self.known_labels = torch.hstack( [
                known_labels,
                1 - known_labels ] )
        else:
            self.known_labels  = known_labels

        self.known_labels = self.known_labels.to( torch.float32 )
    
    def set_desired_label( self, desired_label ):
        if len( desired_label ) <= 1:
            desired_label = torch.hstack( [ 
                1 - desired_label,
                desired_label
            ] )

        self.desired_label = desired_label
    
    def set_bayesian_weights( self ):
        
        w_mu = torch.zeros_like( self.rep( self.dec( self.loc ) ) ).squeeze()
        w_mu = torch.hstack( [ w_mu, torch.ones(1) ] )
        w_cov = torch.eye( w_mu.shape[0] )

        self.w_dist = Gaussian( w_mu, w_cov )
        
    def log_proba( self, w, x ):

        rep_x = torch.hstack( [ self.rep( self.dec( x ) ),
                                torch.ones( 1, 1 ) ] )

        x_likelihood = torch.matmul(
                self.desired_label,
                torch.log_softmax( torch.matmul( w, rep_x.T ), 0 )
        )

        data_likelihood = torch.matmul(
                torch.log_softmax( torch.matmul( w, self.known_data.T ), 0 ),
                self.known_labels
            )
        data_likelihood = torch.trace( data_likelihood )

        w_log_pdf = self.w_dist.log_prob( w ).sum()
        x_log_pdf = self.log_prob( x )

        return x_likelihood + data_likelihood + w_log_pdf + x_log_pdf 
        
    def laplace_apx( self, cf, laplace_iters=10, x_lr=1e-4, x_iters=1000,
                           w_lr=1e-2, w_iters=100 ):
        self.set_bayesian_weights()
        y_shape = self.desired_label.shape[0]

        clf = log_reg(  tol=1e-4, max_iter=1500 )
        clf.fit( 
            self.known_data[:,:-1].numpy(), 
            np.argmax( self.known_labels.numpy(), axis=1 )
        )

        x_ = self.sample([1]).requires_grad_( True )
        w_ = self.w_dist.sample([y_shape]).requires_grad_( True )
        
        optim_x = torch.optim.Adam( (x_, ), lr=x_lr )
        optim_w = torch.optim.LBFGS( (w_, ), lr=w_lr )
        
        w_ = torch.hstack( [ 
            torch.tensor( clf.coef_ ), 
            torch.tensor( clf.intercept_ ).unsqueeze(0) 
            ] )
        w_ = torch.vstack( [ - w_ , w_ ] ).to( torch.float32 )

        for i in range( laplace_iters ):
            
            def closure():
                if torch.is_grad_enabled():
                    optim_w.zero_grad()
                    optim_x.zero_grad()
                output = - self.log_proba( w_, x_ )
                if output.requires_grad:
                    output.backward( )
                return output
            
            print( "------------- Optimizing: X ---------------" )
            for j in range( x_iters ):
                optim_x.zero_grad()
                output = - self.log_proba( w_, x_ )
                output.backward()
                optim_x.step( )
                
                if j%200 == 0:
                    print( f"Iteration {j}; Loss {output.item()}")

        prec = hessian( self.log_proba, ( w_, x_ ) )
        prec_ww = [
                    [
                      prec[0][0][ i, :, j, : ] for j in range( y_shape )
                    ] for i in range( y_shape )
                  ]
        prec_ww = torch.cat( [
            torch.cat( [ col for col in row ], dim=0 ) for row in prec_ww
            ], dim=1 )

        prec_wx = [ prec[0][1][ i, :, 0, : ] for i in range( y_shape ) ]
        prec_wx = torch.cat( [ row for row in prec_wx ], dim=0 )

        prec_xw = [ prec[1][0][ 0, :, i, : ] for i in range( y_shape ) ]
        prec_xw = torch.cat( [ col for col in prec_xw ], dim=1 )

        prec_xx = prec[1][1][ 0, :, 0, : ]
        prec_stack = torch.cat( [
            torch.cat( [ prec_ww, prec_wx ], dim=1 ),
            torch.cat( [ prec_xw, prec_xx ], dim=1 ) ], dim=0 )

        mu_stack = torch.hstack( [ w_.flatten(), x_.flatten() ] )
        cov_stack = torch.inverse( - prec_stack )
 
        return Gaussian( mu_stack, cov_stack )

def cf_posterior( joint_dist, prior, ref_idxs ):
    assert type( joint_dist ) is Gaussian, "Error"
    assert type( prior ) is Gaussian, "Error"

    joint_mu = joint_dist.loc
    joint_cov = joint_dist.covariance_matrix
    joint_prec = joint_dist.precision_matrix

    prior_mu = prior.loc
    prior_cov = prior.covariance_matrix
    prior_prec = prior.precision_matrix

    ref_idxs = joint_dist.verify_idxs( ref_idxs )

    rv_idxs = set( range( joint_mu.shape[0]) ) - set( ref_idxs.tolist() )
    rv_idxs = torch.tensor( list( rv_idxs ), dtype=torch.long )
    rv_idxs = torch.sort( rv_idxs )[0]
    
    marg_mu = joint_mu[ref_idxs]
    marg_cov = joint_cov[ torch.meshgrid( ref_idxs, ref_idxs ) ]
    marg_prec = torch.inverse( marg_cov )
    cond_cov = joint_cov[ torch.meshgrid( ref_idxs, rv_idxs ) ]

    prec_x_ref_given_x = torch.inverse( 
        marg_cov - cond_cov @ marg_prec @ cond_cov.T 
    )

    prec = torch.vstack( [
        torch.hstack( [ 
            prec_x_ref_given_x, 
            - ( prec_x_ref_given_x @ cond_cov @ marg_prec )
        ] ),
        torch.hstack( [
            - ( prec_x_ref_given_x @ cond_cov @ marg_prec ).T, 
            prior_prec + marg_prec @ cond_cov @ prec_x_ref_given_x @ cond_cov @ marg_prec
        ])
    ] )

    mu = torch.inverse( prec ) @ torch.hstack( [
                prec_x_ref_given_x @ ( marg_mu - cond_cov @ marg_prec @ marg_mu ),
                marg_prec @ cond_cov @ prec_x_ref_given_x @ ( cond_cov @ marg_prec @ marg_mu - marg_mu ) +\
                    prior_prec @ prior_mu
    ])
    
    return Gaussian( mu, torch.inverse( prec ) )

