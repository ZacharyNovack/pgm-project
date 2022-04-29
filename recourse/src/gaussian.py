import torch
from torch.distributions import MultivariateNormal


class Gaussian( MultivariateNormal ):
        def __init__( self, mu, cov, round=True, eps=5e-4 ):
            def nearest_PSD( A ):
                A_ = 0.5 * ( A + A.T )
                L, Q = torch.linalg.eigh(A_)
                L = torch.maximum( L, torch.tensor( eps ) )
                B = Q @ torch.diag(L) @ Q.T.conj()
                return B
            
            if round:
                cov = nearest_PSD( cov )
            
            print( 'real', torch.linalg.eigvals( cov ).real.min() )
            print( 'imag', torch.linalg.eigvals( cov ).imag.min() )
            super().__init__( loc=mu, covariance_matrix= cov )
        
        def verify_idxs( self, idxs ):
            if not torch.is_tensor( idxs ):
                idxs = torch.tensor( idxs, dtype=torch.long )
            else:
                idxs = idxs.to( torch.long )
            
            if not torch.is_tensor( idxs ):
                idxs = torch.tensor( idxs, dtype=torch.long )
            else:
                idxs = idxs.to( torch.long )
                
            assert all( idxs >= 0 ), "Index cannot be negative"

            if len( idxs ) > self.loc.shape[0]:
                raise ValueError( "Dimension Mismatch. Input, indexes" +
                                 " has more dimensions than distribution mean.")            
            elif any( idxs > self.loc.shape[0] ):
                raise ValueError( f"Dimension Mismatch. Distribution has maximum dimension {mu.shape[0]}." +
                                f" indexes is requesting dimension: {max(idxs)}" ) 
            
            #idxs = torch.sort( idxs )
            #return idxs.values
            return idxs
        
        def conditioned_on( self, x_B, conditional_idxs ):
            mu = self.loc
            cov = self.covariance_matrix
              
            conditional_idxs = self.verify_idxs( conditional_idxs )
            
            rv_idxs = set( range( mu.shape[0]) ) - set( conditional_idxs.tolist() )
            rv_idxs = torch.tensor( list( rv_idxs ), dtype=torch.long )
            rv_idxs = torch.sort( rv_idxs )[0]
            
            mu_A = mu[ rv_idxs ]
            mu_B = mu[ conditional_idxs ]
            
            cov_AA = cov[ torch.meshgrid( rv_idxs, rv_idxs ) ]
            cov_AB = cov[ torch.meshgrid( rv_idxs, conditional_idxs ) ]
            cov_BA = cov[ torch.meshgrid( conditional_idxs, rv_idxs ) ]
            cov_BB = cov[ torch.meshgrid( conditional_idxs, conditional_idxs ) ]
            
            conditional_mu = mu_A + cov_AB @ torch.inverse( cov_BB ) @ ( x_B - mu_B )
            conditional_cov = cov_AA - cov_AB @ torch.inverse( cov_BB ) @ cov_BA 
            
            return Gaussian( conditional_mu, conditional_cov )
            
        def marginal_on( self, idxs ):
            mu = self.loc
            cov = self.covariance_matrix
            
            idxs = self.verify_idxs( idxs )
            
            marginal_mu = mu[ idxs ]
            marginal_cov = cov[ torch.meshgrid( idxs, idxs ) ]
            return Gaussian( marginal_mu, marginal_cov )
        
        def marginalize_over( self, idxs ):
            idxs = self.verify_idxs( idxs )
            
            mu = self.loc
            cov = self.covariance_matrix
            
            rv_idxs = set( range( mu.shape[0]) ) - set( idxs.tolist() )
            rv_idxs = torch.tensor( list( rv_idxs ), dtype=torch.long )
            rv_idxs = torch.sort( rv_idxs )[0]

            mu_A = mu[ rv_idxs ]

            cov_AA = cov[ torch.meshgrid( rv_idxs, rv_idxs ) ]
            u, v = torch.linalg.eigh( cov_AA )

            return Gaussian( mu_A, cov_AA )
        
        def linear_conditional( self, A, b, residual_cov ):
            mu = self.loc
            cov = self.covariance_matrix
            
            if not torch.is_tensor( A ):
                A = torch.tensor( A, dtype=mu.dtype )
            if not torch.is_tensor( b ):
                b = torch.tensor( b, dtype=mu.dtype )
            if not torch.is_tensor( residual_cov ):
                residual_cov = torch.tensor( residual_cov, dtype=mu.dtype )
                
            prod_mu = torch.hstack( [ mu, A.T @ mu + b ] )
            prod_cov = torch.vstack( [
                torch.hstack( [ cov, cov @ A ] ),
                torch.hstack( [ A.T @ cov, residual_cov +  A.T @ cov @ A ] ),
            ])
            return Gaussian( prod_mu, prod_cov )

