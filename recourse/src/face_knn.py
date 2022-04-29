import torch
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.generic import shortest_path_length


def condition_matrix( X, immutable_idxs=[], non_decreasing_idxs=[],
                      non_increasing_idxs=[] ):
    K = torch.ones( ( X.shape[0], X.shape[0] ) ).to( bool )

    #Handle immutable idxs (Distance matrix of cols must be zero)
    X_ = X[:,immutable_idxs]
    r = ( X_**2 ).sum( axis=1 ).reshape( -1, 1 )
    D = r - 2 * X_ @ X_.T + r.T
    K = K & torch.logical_not( torch.maximum( D, torch.zeros(1) ).to( bool ) )

    #Handle non-decreasing idxs
    for col in non_decreasing_idxs:
        x_ = X[:,col].reshape( -1, 1 )
        K = K & ( ( x_ @ x_.T ) <= x_**2 ).T

    #Handle non-increasing idxs (outer product)
    for col in non_increasing_idxs:
        x_ = X[:,col].reshape( -1, 1 )
        K = K & ( ( x_ @ x_.T ) >= x_**2 ).T

    return K

def get_weights_KNN( X, distance_threshold=np.inf, 
                    n_neighbors=20,
                    cov = None,
                    weight_function = lambda x: - torch.log( x ),
                    immutable_idxs=[], non_decreasing_idxs=[],
                    non_increasing_idxs=[]  ):
    # Calculate distance matrix of inputs
    print( "\t Creating Distance Matrix...")
    if cov is None:
        cov = torch.eye( X.shape[1] )

    L = torch.linalg.cholesky( cov, upper=True )
    X_ = X @ torch.linalg.inv( L )

    r = ( X_**2 ).sum( axis=1 ).reshape( -1, 1 )
    K = r - 2 * X_ @ X_.T + r.T
    K = torch.sqrt( torch.maximum( K, torch.zeros(1) ) )
    #Throw away points whose distance is above a threshold
    K[ K > distance_threshold ] = 0

    #Only keep the closest n+1 neighbors
    n_samples = X.shape[0]
    print( "\t Sorting Matrix...")
    t = K.numpy().argsort( axis=1 )[:,n_neighbors+1:]
    t = torch.tensor( t )

    print( "\t Keep only {} nearest neighbors...".format( n_neighbors ) )
    K[ torch.arange( K.shape[0] ).unsqueeze(1), t ] = 0
    # zero-out everywhere that does not follow the conditions
    cond_mat = condition_matrix( 
        X, 
        immutable_idxs = immutable_idxs,
        non_decreasing_idxs = non_decreasing_idxs,
        non_increasing_idxs = non_increasing_idxs 
    )
    print( "\t Finding Condition Matrix..." )
    K = K * cond_mat
    #Calculate weight of remaining nodes
    K[ K != 0 ] = weight_function( K[ K != 0 ] )

    return torch.maximum( K, torch.zeros(1) ) #handle floating-point errors

def create_graph( X, distance_threshold=np.inf,
                  n_neighbors=20,
                  cov = None,
                  weight_function = lambda x: - torch.log( x ),
                  immutable_idxs=[], non_decreasing_idxs=[],
                  non_increasing_idxs=[] ):
    
    G = nx.Graph()
    G.add_nodes_from( np.arange( X.shape[0] ) )
    print( "Creating Weight Kernel...")
    kernel = get_weights_KNN(        
        X, 
        distance_threshold = distance_threshold,
        n_neighbors = n_neighbors,
        cov = cov,
        weight_function = weight_function,
        immutable_idxs = immutable_idxs,
        non_decreasing_idxs = non_decreasing_idxs,
        non_increasing_idxs = non_increasing_idxs 
    )

    print( "Creating Graph...")
    idxs = np.vstack( np.where( kernel > 0 ) )
    w_edges = np.vstack( [ idxs, kernel[ idxs[0], idxs[1] ] ] )
    G.add_weighted_edges_from( w_edges.T )  

    return G
    

def get_counterfactuals(X, reference, desired_output, classifier,
                    decoder=torch.nn.Identity(), 
                    num_counterfactuals=3, n_neighbors=20,
                    distance_threshold=np.inf, 
                    cov = None,
                    weight_function = lambda x: - torch.log( x ), 
                    immutable_idxs=[],
                    non_decreasing_idxs=[], non_increasing_idxs=[], graph=None ):
    """

        Args:
            - reference: tuple of (sample_x, y) to use as reference for generating counterfactuals,
    """
    if graph is None:
        G = create_graph( 
            X,     
            n_neighbors = n_neighbors,
            distance_threshold = distance_threshold,
            cov = cov,
            weight_function = weight_function,     
            immutable_idxs = immutable_idxs,
            non_decreasing_idxs = non_decreasing_idxs,
            non_increasing_idxs = non_increasing_idxs 
        )
    else:
        G = graph

    print( "Finding Shortest Paths..." )
    min_dist = np.inf
    path = shortest_path( G, source=reference[2] )
    dist = shortest_path_length( G, source=reference[2] )
    C_x = []
    C_dist = []

    predictions = classifier( X )
    for key, val in path.items():
        key = int( key )
        if ( predictions[ key, desired_output.to( bool ) ] >= 0.55 ):
            C_x.append( key )
            C_dist.append( dist[key] )

    print( "Returning Counterfactuals..." )
    C = X[ sorted( C_x, key = lambda t: dist[t] )[:num_counterfactuals], : ]
    print( C.shape )
    return decoder( C ).detach(), G
