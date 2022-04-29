import os

import pandas as pd
import numpy as np
import networkx as nx

import torch
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.linear_model import LinearRegression

from collections import OrderedDict

#Setup Dataset
class Features:
    def __init__( self, X, col_name, softmax_temp=0.75, 
                  categorical=False, immutable=False, nonactionable=False,
                  log=False ):
        self.col = col_name
        self.categorical = categorical
        self.immutable = immutable
        self.nonactionable = nonactionable
        self.log = log

        self.softmax = torch.nn.Softmax( dim=1 )
        self.softmax_temp = softmax_temp

        if self.categorical:
            self.categorical_encoding( X )
        elif log:
            self.encoding = lambda x: torch.tensor( 
                np.vstack( [ x.astype( np.float32 ) ] ) ).log().T
            self.decoding = lambda y: np.exp( np.array( y, ndmin=2, dtype=np.float64 ) )
            self.num_classes = 1
            self.classes = None       
        else:
            self.encoding = lambda x: torch.tensor( 
                np.vstack( [ x.astype( np.float32 ) ] ) ).T
            self.decoding = lambda y: y
            self.num_classes = 1
            self.classes = None

        self.mu = self.mean( X )
        self.var = self.variance( X )

    def categorical_encoding( self, X ):
        classes, counts = np.unique( X, return_counts=True )

        num_classes = len( classes )
        encoding = lambda x: torch.tensor( 
            [ [ ( x_ == class_ ) for class_ in classes ] for x_ in x ],
            dtype = torch.int32 
        )
        decoding = lambda y: np.array( 
            [ classes[ np.argmax( y_ ) ] for y_ in y ], ndmin=2 
        ).T

        self.encoding = encoding
        self.decoding = decoding
        self.num_classes = num_classes
        self.classes = classes
        self.class_probs = torch.tensor( counts / X.shape[0], dtype=torch.float32 )

    def __call__( self, x ):
        return self.encoding( x )

    def decode( self, y ):
        return self.decoding( y )

    def sample( self, x ):
        if not self.categorical:
            return x

        x = self.softmax( x / self.softmax_temp )

        return x

    def mean( self, X ):
        if self.categorical:
            logit = torch.log( self.class_probs / ( 1 - self.class_probs ) )
            return logit
        elif self.log:
            return torch.mean( torch.tensor(X, dtype=torch.float32 ).log(), axis=0 ).unsqueeze(0)
        else:
            return torch.mean( torch.tensor(X, dtype=torch.float32 ), axis=0 ).unsqueeze(0)

    def variance( self, X ):
        if self.categorical:
            return torch.ones( self.num_classes )
        elif self.log:
            return torch.var( torch.tensor(X, dtype=torch.float32 ).log(), axis=0 ).unsqueeze(0)
        else:
            return torch.var( torch.tensor(X, dtype=torch.float32 ), axis=0 ).unsqueeze(0)
    

#Setup Dataset
class CSV(Dataset):
    def __init__( self, csv_filename, 
                  target_column=None, 
                  softmax_temp=0.75, 
                  drop_columns = None,
                  transform=None, 
                  set_na =[],
                  immutable_columns = [], 
                  nonactionable_columns = [],
                  log_transform = [], 
                  dataset="LUCAS"):

        self.frame = pd.read_csv( csv_filename ) 
        self.frame = self.frame.drop( drop_columns, axis=1 )
        self.frame = self.frame.replace( set_na, np.nan )
        self.frame = self.frame.dropna( axis=0 )
        if 'adult' in csv_filename.lower():
            self.protect_list = self.frame['sex'].to_numpy() == ' Female'
            self.frame = self.frame.drop(['sex'], axis=1 )
            drop_columns.append('sex')
        elif 'german' in csv_filename.lower():
            self.protect_list = self.frame['Foreign worker'].to_numpy() == 'yes'
            self.frame = self.frame.drop(['Foreign worker'], axis=1 )
            drop_columns.append('Foreign worker')
        elif 'ssl' in csv_filename.lower():
            self.protect_list = self.frame['RACE CODE CD'].to_numpy() == 'nw'
            self.frame = self.frame.drop(columns=['RACE CODE CD'], axis=1 )
            drop_columns.append('RACE CODE CD')
        elif 'lipton' in csv_filename.lower():
            self.protect_list = self.frame['gender'].to_numpy() == 0
            self.frame = self.frame.drop(['gender'], axis=1 )
            drop_columns.append('gender')
        self.columns = self.frame.columns
        features = {}

        for col in self.columns:
            if col in immutable_columns:
                immutable = True
                nonactionable=False
            elif col in nonactionable_columns:
                nonactionable=True 
                immutable = False
            else:                
                immutable = False
                nonactionable=False


            if self.frame[col].dtype == "O":
                features[ col ] = Features( self.frame[col].to_numpy(), 
                                                col,
                                                softmax_temp=softmax_temp, 
                                                categorical=True,
                                                immutable=immutable,
                                                nonactionable=nonactionable 
                )
            else:
                features[ col ] = Features( self.frame[col].to_numpy(), 
                                                col,
                                                softmax_temp=softmax_temp, 
                                                categorical=False,
                                                immutable=immutable,
                                                nonactionable=nonactionable,
                                                log = col in log_transform
                )
        self.features = OrderedDict( features )
        self.target_column = target_column
        self.label_idxs = np.where( 
                    [ col in self.target_column for col in self.frame.columns ] 
                )[0]

        self.drop_cols = np.where( [ drop_ in self.frame.columns for drop_ in self.target_column ] )
        self.drop_cols = np.unique( self.drop_cols[0] )

    def __len__( self ):
        return len( self.frame )

    def __getitem__( self, idx ):
        if type( idx ) is int:
            idx = [idx]

        rows = self.frame.iloc[idx].to_numpy()

        x, y = self.encode( rows )

        sample = [ x, y ]
        
        return sample

    def where_label( self, labels ):
        idx = 0
        return_labels = {}
        for key in self.features.keys():
            feature = self.features[ key ]
            if key in labels:
                return_labels[ key ] = list( range( idx, idx + feature.num_classes ) )
            idx = idx + feature.num_classes

        return return_labels

    def mean( self ):
        mu = torch.empty( 0 )
        for key in self.features.keys():
            if key == self.target_column:
                continue
            mu = torch.hstack( [ mu, self.features[key].mu ] )

        return mu

    def covariance( self, feasible=False ):
        var = torch.empty( 0 )
        sig = torch.empty( 0 )
        for key in self.features.keys():
            feature = self.features[ key ]
            if key == self.target_column:
                continue
            var = torch.hstack( [ var, feature.var ] )
            if feature.immutable:
                sig = torch.hstack( [ sig, torch.zeros_like( feature.var ) ] )
            else:
                sig = torch.hstack( [ sig, torch.ones_like( feature.var ) ] )

        cov = torch.diag( var )
        if feasible:
            return torch.outer( sig, sig ) * cov
        else:
            return cov  

    def nonactionable_params( self, mu, cov ):     
        return mu, cov 

    def encode( self, x ):
        assert x.shape[1] == len( self.features ), \
            f"Pandas frame has {self.frame.shape[1]} features. Input has {x.shape[1]} features."

        x_ = torch.empty(x.shape[0],0)
        y_ = torch.empty(x.shape[0],0)
        for col_data, key in zip( x.T, self.features.keys() ):
            feature = self.features[ key ]
            if feature.col in self.target_column:
                y_ = torch.hstack( [ y_, feature( col_data ) ] )
            else:
                x_ = torch.hstack( [ x_, feature( col_data ) ] )

        #x_.requires_grad_( True )
        return x_, y_

    def normalize( self, x ):
        if len( x.shape ) == 1:
            x = x.unsqueeze(0)

        x_ = torch.empty( x.shape[0], 0 )
        y_ = torch.empty( x.shape[0], 0 )

        idx = 0
        for key in self.features.keys():
            feature = self.features[ key ]
            if x[:,idx:idx + feature.num_classes].shape[1] == 0:
                continue

            if feature.col in self.target_column:
                y_ = torch.hstack( [ y_, feature.sample( x[:,idx:idx + feature.num_classes] ) ] )
            else:
                x_ = torch.hstack( [ x_, feature.sample( x[:,idx:idx + feature.num_classes] ) ] )

            idx += feature.num_classes

        if len( y_ ) == 0:
            y = None

        return x_, y_

    def decode( self, x, sample=False ):
        # if len( x.shape ) == 1:
        #     x = x.unsqueeze(0)
            
        x_ = np.empty( ( x.shape[0], 0 ) )
        y_ = np.empty( ( x.shape[0], 0 ) )

        idx = 0
        for key in self.features.keys():
            feature = self.features[ key ]
            if x[:,idx:idx + feature.num_classes].shape[1] == 0:
                continue

            if feature.col in self.target_column:
                y_ = np.hstack( [ y_, feature.decode( x[:,idx:idx + feature.num_classes] ) ] )
            else:
                x_ = np.hstack( [ x_, feature.decode( x[:,idx:idx + feature.num_classes] ) ] )

            idx += feature.num_classes

        if len( y_ ) == 0:
            y = None

        return x_, y_

    def drop_col( self, col ):
        self.frame.drop( col, axis=1, inplace=True )

    def reindex( self, labels ):
        labels = list( labels ) + list( set( self.columns ) - set( labels ) ) 
        self.frame = self.frame.reindex(columns=labels)
        self.frame = self.frame.dropna( axis=1 )
        self.columns = self.frame.columns

        self.label_idxs = np.where( 
                [ col in self.target_column for col in self.columns ] 
            )[0]
        
        for label in reversed( labels ):
            self.features.move_to_end( label, last=False )

    def as_pd( self, x, y=None ):
        cols = self.columns
        if y is not None:
            z = np.hstack( [ x, y ] )
        else:
            z = np.hstack( [ x ] )
            cols = cols.drop( self.columns[self.label_idxs] )

        if len( z.shape ) < 2:
            return pd.DataFrame( np.array( z ).reshape( 1, -1 ), columns=cols )
        else:
            return pd.DataFrame( z, columns=cols )

class CausalCSV(CSV):
    def __init__( self, csv_fname, dag_fname, target_column=None,
                  softmax_temp=0.75, drop_columns=None, set_na =[], delim=" -> ", 
                  transform=None, dataset=None,
                  immutable_columns = [], 
                  nonactionable_columns = [],
                  log_transform = [],  ):
        super().__init__( csv_fname, target_column=target_column, softmax_temp=0.75, 
                          drop_columns=drop_columns, transform=transform, 
                          set_na =set_na, dataset=dataset,
                          immutable_columns = immutable_columns, 
                          nonactionable_columns = nonactionable_columns,
                          log_transform = log_transform  )
        
        #Read edges from dag_fname into a networkx graph
        self.G = nx.read_edgelist( dag_fname,
                              create_using=nx.DiGraph,
                              delimiter=delim )

        #Include all nodes without edges in dag_fname into the graph
        self.G.update( nodes= set( self.columns ) - set( self.G.nodes ) )
        for col in drop_columns:
            if col in self.G.nodes:
                self.G.remove_node( col )
        
        target_column = self.target_column
        if target_column is not None:
            if type( target_column ) is not list:
                target_column = [ target_column ]
            
            for col in target_column:
                if col in self.G.nodes:
                    self.G.remove_node( col )
        
        sorted_nodes = [ node for node, w in sorted( self.G.in_degree, key=lambda x: x[1] ) ]
        for i, node in enumerate( sorted_nodes ):
            parents = list( self.G.reverse()[ node ].keys() )
            for parent in parents:
                if parent in sorted_nodes[i:]:
                    j = np.where( [ parent == n for n in sorted_nodes ] )[0]
                    j = j.item()
                    sorted_nodes[i], sorted_nodes[j] = sorted_nodes[j], sorted_nodes[i]

        self.reindex( sorted_nodes )
        mu, cov = self.gaussian_prior()

        self.mu = mu
        self.cov = cov

        def mean( self ):
            return self.mu
        
        def covariance( self, feasible=False ):
            if not feasible:
                cov = self.cov
            else:
                _, cov = self.gaussian_prior( feasible=True )

            return cov

    def drop_node( self, col ):
        self.G.remove_node( col )
        self.drop_col( col )

    def gaussian_prior( self, feasible=False ):
        mu = torch.zeros((0,))
        cov = torch.zeros((0,0))
        sig = torch.empty( 0 )
        feature_idxs = {}            
        idxs = 0

        for key in self.features.keys():
            if key == self.target_column:
                continue

            feature = self.features[ key ]
            if feature.immutable:
                sig = torch.hstack( [ sig, torch.zeros_like( feature.var ) ] )
            else:
                sig = torch.hstack( [ sig, torch.ones_like( feature.var ) ] )

            parents = list( self.G.predecessors( key ) )
            feature_idxs[feature.col] = range( 
                    idxs, 
                    idxs + np.array( feature.mu, ndmin=1 ).shape[0] 
                )

            idxs += np.array( feature.mu, ndmin=1 ).shape[0]

            print( key, parents )
            if len( parents ) == 0:

                mu = torch.hstack( [ 
                    mu, 
                    feature.mu
                ] )
                cov_xx = cov
                cov_yy = torch.diag( feature.var )
                cov_xy = torch.zeros( ( cov_xx.shape[0], cov_yy.shape[0] ) )
                cov_yx = torch.zeros( ( cov_yy.shape[0], cov_xx.shape[0] ) )
                cov = torch.vstack( [ 
                    torch.hstack( [ cov_xx, cov_xy ] ),
                    torch.hstack( [ cov_yx, cov_yy ] )
                ] )
            else:
                ranges = np.hstack( 
                    [ list( feature_idxs[ parent ] ) for parent in parents ]
                )

                parent_mu = mu[ ranges ]
                parent_cov = cov[ tuple( np.meshgrid( ranges, ranges ) ) ]
                
                X = self[ : ][ 0 ][ :, ranges  ]
                y = self[ : ][ 0 ][ :, feature_idxs[ feature.col ] ]

                clf = LinearRegression( fit_intercept=True )
                clf.fit( X.detach(), y.detach() )
                
                u, s, vh = np.linalg.svd( clf.coef_, full_matrices=False )
                norm = 0.25 * np.linalg.norm( s )
                s =  s / norm

                A = torch.tensor( u @ np.diag( s ) @ vh )
                b = torch.tensor( clf.intercept_ / norm )

                child_mu = feature.mu
                child_cov = torch.diag( feature.var )
                
                joint_cov_xx = parent_cov
                joint_cov_xy = parent_cov @ A.T
                joint_cov_yx = A @ parent_cov
                joint_cov_yy = child_cov + A @ parent_cov @ A.T
                
                joint_mu = torch.hstack( [ parent_mu, A @ parent_mu + b ] )

                joint_cov = torch.vstack( [ 
                    torch.hstack( [ joint_cov_xx, joint_cov_xy ] ),
                    torch.hstack( [ joint_cov_yx, joint_cov_yy ] )
                ] )
                joint_ranges = np.hstack( [ ranges, list( feature_idxs[ key ] ) ] )
                
                mu = torch.hstack( [ mu, child_mu ] )
                cov = torch.vstack( [ 
                    torch.hstack( [ 
                        cov, 
                        torch.zeros( ( cov.shape[0], child_cov.shape[0] ) ) ] ),
                    torch.hstack( [ 
                        torch.zeros( ( child_cov.shape[0], cov.shape[0] ) ), 
                        torch.zeros( child_cov.shape ) ] )
                ] )

                mu[ joint_ranges ] = joint_mu
                cov[ tuple( np.meshgrid( joint_ranges, joint_ranges ) ) ] = joint_cov
        
        if feasible:
            return mu, torch.outer( sig ) * cov
        else:
            return mu, cov

    def nonactionable_params( self, mu, cov ):
        feature_idxs = {}   
        idxs = 0

        for key in self.features.keys():
            feature = self.features[key]

            feature_idxs[feature.col] = range( 
                    idxs, 
                    idxs + np.array( feature.mu, ndmin=1 ).shape[0] 
                )

            idxs += np.array( feature.mu, ndmin=1 ).shape[0]

            if not feature.nonactionable:
                continue

            parents = list( self.G.predecessors( key ) )
            if len( parents ) == 0:
                continue

            ranges = np.hstack( 
                    [ list( feature_idxs[ parent ] ) for parent in parents ]
            )

            parent_mu = mu[ ranges ]
            parent_cov = cov[ tuple( np.meshgrid( ranges, ranges ) ) ]

            X = self[ : ][ 0 ][ :, ranges  ]
            y = self[ : ][ 0 ][ :, feature_idxs[ feature.col ] ]
            
            clf = LinearRegression( fit_intercept=True )
            clf.fit( X.detach(), y.detach() )
                
            u, s, vh = np.linalg.svd( clf.coef_, full_matrices=False )
            norm = 0.25 * np.linalg.norm( s )
            s = s / norm

            A = torch.tensor( u @ np.diag( s ) @ vh )
            b = torch.tensor( clf.intercept_ / norm )

            child_range = feature_idxs[feature.col]
            child_mu = mu[ child_range ]
            child_cov = cov[ tuple( np.meshgrid( child_range, child_range ) ) ]

            joint_cov_xx = parent_cov
            joint_cov_xy = parent_cov @ A.T
            joint_cov_yx = A @ parent_cov
            joint_cov_yy = child_cov + A @ parent_cov @ A.T
                
            joint_mu = torch.hstack( [ parent_mu, A @ parent_mu + b ] )
            joint_cov = torch.vstack( [ 
                    torch.hstack( [ joint_cov_xx, joint_cov_xy ] ),
                    torch.hstack( [ joint_cov_yx, joint_cov_yy ] )
            ] )
            joint_ranges = np.hstack( [ ranges, list( feature_idxs[ key ] ) ] )

            mu[ joint_ranges ] = joint_mu
            cov[ tuple( np.meshgrid( joint_ranges, joint_ranges ) ) ] = joint_cov

        return mu, cov


class TorchDataset(Dataset):
    def __init__( self, dataset ):
        self.data = dataset.transform( dataset.data )
        self.targets = dataset.targets

    def __getitem__( self, idx ):
        return ( self.data[idx], self.targets[idx] )

    def __len__( self ):
        return len( self.data )

    def decode( self, x, sample=False ):
        return x, None

    def encode( self, x ):
        return x, None

def _make_dataset( fname="../data/datasets/lucas/lucas0_train.csv",
                   dag_fname = None,
                   target_column="",
                   softmax_temp=0.75,
                   drop_columns=None,
                   set_na=[],
                   immutable_columns = [],
                   nonactionable_columns = [],
                   log_transform = [],
                   dataset="LUCAS",
                   onehot = False ):

    if dataset.lower() == "mnist":
        data = torchvision.datasets.MNIST( fname, train=True, download=True,
			    transform=torchvision.transforms.Compose([
				lambda x: x > 0,
				lambda x: x.to( torch.float32 )
			    ]))
        idxs = ( data.targets == 4 ) | \
               ( data.targets == 9 )
        
        data.data = data.data[ idxs ]
        targets = data.targets[idxs]
        if onehot:
            targets = torch.nn.functional.one_hot( targets )
            targets = targets[ targets.sum( axis=-1 ) != 0 ]
        
        data.targets = targets
        data = TorchDataset( data )   
        
        return data
        
    if not os.path.exists( dag_fname ):
        print( "No DAG file provided/DAG path does not exist" )
        data = CSV( fname, target_column=target_column,
                softmax_temp=softmax_temp, dataset=dataset,
                drop_columns=drop_columns, set_na=set_na,
                immutable_columns = immutable_columns, 
                nonactionable_columns = nonactionable_columns,
                log_transform = log_transform  )
    else:
        data = CausalCSV( fname, dag_fname, target_column=target_column, 
                softmax_temp=softmax_temp, dataset=dataset,
                drop_columns=drop_columns, set_na=set_na,
                immutable_columns = immutable_columns, 
                nonactionable_columns = nonactionable_columns,
                log_transform = log_transform  )

    return data
