import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import numpy as np

class LogisticRegression( nn.Module ):
    def __init__( self ):
        super( LogisticRegression, self ).__init__()
        self.fc1 = nn.Linear( 1, 1 )
        self.sigmoid = torch.nn.Sigmoid()

class Sklearn_NN( nn.Module ):
    def __init__( self, num_hidden=3 ):
        super( Sklearn_NN, self ).__init__()
        self.num_hidden = num_hidden
        self.layers = []
        for layer in range( num_hidden + 1 ):
            self.layers.append( 
                torch.nn.Linear( 1, 1 )
            )

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def create_model( self, data, labels, out_folder=None ):
        X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.2, random_state=0 )

        print( y_test.sum() )
        hidden_layer_sizes = [ 50, ] + [ 20, ] * ( self.num_hidden - 1 )
        clf = MLPClassifier( hidden_layer_sizes=hidden_layer_sizes, max_iter=2000, 
                             batch_size=32, early_stopping=True )
        clf.fit( X_train, y_train )
        print( "Train Accuracy : {}".format( clf.score( X_train, y_train ) ) )
        print( "Test Accuracy : {}".format( clf.score( X_test, y_test ) ) )

        for i in range( self.num_hidden + 1 ):
            layer = torch.nn.Linear( clf.coefs_[i].shape[0], clf.coefs_[i].shape[1] )
            for param in layer.named_parameters():
                coef = clf.coefs_[i]
                bias = clf.intercepts_[i]

                param[1].requires_grad_( False )
                if "weight" in param[0]:
                    param[1].copy_( torch.tensor( coef.T, dtype=torch.float32 ) )
                elif "bias" in param[0]:
                    param[1].copy_( torch.tensor( bias, dtype=torch.float32 ) )
            self.layers[i] = layer

        if out_folder is not None:
            torch.save( self, out_folder )

    def forward( self, x, rep=False ):
        x = x.view( -1, x.shape[-1] )
        for i in range( self.num_hidden ):
            x = self.layers[i]( x )
            x = self.relu( x )

        if rep is True:
            return x
        
        x = self.layers[-1]( x )
        x = torch.hstack( [ self.sigmoid( - x ), self.sigmoid( x ) ] )
        return x

        
class LogisticRegression( nn.Module ):
    def __init__( self ):
        super( LogisticRegression, self ).__init__()
        self.fc1 = nn.Linear( 1, 1 )
        self.sigmoid = torch.nn.Sigmoid()

    def create_model( self, data, labels, out_folder=None ):
        clf = linear_model.LogisticRegression( C=1, tol=1e-4, max_iter=1000, fit_intercept=True )
        clf.fit( data, labels )

        coef = clf.coef_
        bias = clf.intercept_
        
        self.fc1 = nn.Linear( coef.flatten().shape[0], 1 )
        for param in self.named_parameters():
            param[1].requires_grad_( False )
            if "weight" in param[0]:
                param[1].copy_( torch.tensor( coef, dtype=torch.float32 ) )
            elif "bias" in param[0]:
                param[1].copy_( torch.tensor( bias, dtype=torch.float32 ) )

            param[1].requires_grad_( True )

        if out_folder is not None:
            torch.save( self, out_folder )

    def forward( self, x, rep=False ):
        x = x.view( -1, x.shape[-1] )
        if rep:
            return x

        x = self.fc1( x )
        x = torch.hstack( [ self.sigmoid( - x ), self.sigmoid( x ) ] )
        return x

class MnistVAE(nn.Module):
    def __init__( self ):
        super( MnistVAE, self ).__init__()
        self.latent = 2

        self.fc1 = nn.Linear( 784, 784 )
        self.fc2 = nn.Linear( 784, 784 )
        self.fc3 = nn.Linear( 784, self.latent )
        self.fc4 = nn.Linear( 784, self.latent )
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

        self.fc5 = nn.Linear( self.latent, 784 )
        self.fc6 = nn.Linear( 784, 784 )
        self.fc7 = nn.Linear( 784, 784 )
        self.drop5 = nn.Dropout()
        self.drop6 = nn.Dropout()
        self.softplus = nn.Softplus()

    def encoder( self, x ):

        x = x.view( -1, 784 )
        x = F.relu( self.drop2( self.fc2( x ) ) )

        mn = self.fc3( x )
        sd = self.fc4( x )
        eps = torch.normal( torch.zeros( x.shape[0], self.latent ) )

        z = mn + torch.exp( sd ) * eps
        return z

    def decoder( self, x ):
        x = x.view( -1, self.latent )
        x = F.relu( self.drop5( self.fc5( x ) ) )
        x = torch.sigmoid( self.fc7( x ) )

        x = x.view( -1, 1, 28, 28 )
        return x

    def forward( self, x ):
        z, mn, sd = self.encoder( x )
        x = self.decoder( z )

        return x, mn, sd

class MnistClassifier( nn.Module ):
    def __init__( self ):
        super( MnistClassifier, self ).__init__()
        self.conv1 = nn.Conv2d( 1, 10, kernel_size=5 )
        self.conv2 = nn.Conv2d( 10, 20, kernel_size=5 )
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear( 320, 50 )
        self.fc2 = nn.Linear( 50, 2 )

    def forward( self, x, rep=False ):

        if len( x.shape ) < 4:
            x = x.view( -1, 1, 28, 28 )

        x = F.relu( F.max_pool2d( self.conv1( x ), 2 ) )
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2( x ) ), 2 ) )
        x = x.view( -1, 320 )
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, training=self.training )
        if not rep:
            x = self.fc2( x )
            return F.softmax( x, dim=-1 )
        else:
            return x
