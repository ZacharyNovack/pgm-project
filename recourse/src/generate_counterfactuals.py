import torch
import torch.nn as nn

from sklearn.neural_network import MLPClassifier

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

import configparser
import argparse
import textwrap

from model_utils import *
from data_utils import _make_dataset
from gaussian import Gaussian

import dice
import gradual_construction
from bayes_cf import *
import face_knn

def save_args( fname ):
    """
        Create a config file. Defaults to uci-adult dataset config
        See README.md for more information

        Parameters
        ----------
        fname : str
            Output the configuration to this file

        Returns
        -------
        None
    """

    config = configparser.ConfigParser()
    config['Basic'] = {'dataset': "Adult",
                       'dataset_path': "../data/datasets/adult/adult.data",
                       'dag_path': "../data/datasets/adult/adult.dag",
                       'explainer': "Bayes",
                       'num_counterfactuals': "100",
                       'out_folder': "./adult"}
    
    config['Classifier'] = {"classifier": "Logistic_Regression",
                            "desired_output": "[ 0.0, 1.0 ]",
                            "loss": "BCE",
                            "model_path": "None"}
    
    config['Data'] = {'target_column':"income",
                      'drop_columns': "[ 'capital-gain', 'capital-loss', 'fnlwgt', 'education-num' ]",
                      'set_na': "[ ' ?' ]",
                      'immutable_columns': "[ 'race', 'sex' ]",
                      'non_decreasing_columns': "[ 'age' ]",
                      'non_increasing_columns': "[ ]",
                      'nonactionable_columns': "[ ]",
                      'log_transform': "[ ]",
                      'softmax_temp': "0.6"}
    
    config['Vae'] = {'model_path': "None" }

    config['DiCE'] = {'lr': "1e-2",
                      'iters': "5000",
                      'prox_reg': "1e-2",
                      'dpp_reg': "2"}

    config['Bayes'] = {'alpha': "0.0",
                       'laplace_iters': "1",
                       'x_lr': "1e-2",
                       'x_iters': "5000"}

    config['GradualConstruction'] = { 'lr': "1e-2",
                       'iters': "500",
                       'prox_reg': "1.",
                       'dpp_reg': "5.",
                       'accuracy': "0.95" }

    config['FACE'] = { 'method': "KNN",
                       'n_neighbors': "20",
                       'distance_threshold': "inf" }
                
    with open( fname, "w" ) as configfile:
        config.write( configfile )

def parse_config( fname ):
    """
        Read in a configuration from a known config file

        Parameters
        ----------
        fname: str
            File from which we read in a configuration

        Returns
        -------
        config: configparser.ConfigParser
            configuration that will be used to generate counterfactuals
            Dictionary-like object
    """
    config = configparser.ConfigParser()
    config.read( fname )

    return config

def parse_args(args):
    """
        Parse command line arguments using the argument parser.

        Parameters
        ----------
        args: list
            system arguments passed into the program

        Returns
        -------
        config: configparser.ConfigParser
            configuration that will be used to generate counterfactuals
            Dictionary-like object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument( "--new_config", type=str, default=None )
    parser.add_argument( "--config", type=str, default=None )

    args = parser.parse_args(args)
    if args.new_config is not None:
        #Create a new default configuration
        save_args( args.new_config )
        config = parse_config( args.new_config )
    elif args.config is not None:
        #Use a known configuration file
        config = parse_config( args.config )
    else:
        raise ValueError( textwrap.fill(textwrap.dedent("""
                Must specify a config file. Use --help argument
                for more information.
        """)))

    return config

if __name__=="__main__":

    config = parse_args(sys.argv[1:])
    basic_config = config['Basic']
    classifier_config = config['Classifier']
    data_config = config['Data']
    vae_config = config['Vae']
    dice_config = config['DiCE']
    bayes_config = config['Bayes']
    grad_con_config = config['GradualConstruction']
    face_config = config['FACE']

    # Get dataset
    data = _make_dataset(
            fname=basic_config['dataset_path'],
            dag_fname = basic_config['dag_path'],
            softmax_temp = float( data_config['softmax_temp'] ),
            target_column=data_config['target_column'],
            drop_columns=eval( data_config['drop_columns'] ),
            set_na = eval( data_config['set_na'] ),
            immutable_columns = eval( data_config['immutable_columns'] ),
            nonactionable_columns = eval( data_config[ 'nonactionable_columns' ] ),
            log_transform = eval( data_config[ 'log_transform' ] ),
            dataset=basic_config['dataset'],
    )

    X = data[:][0] #Get preprocessed data
    y = data[:][1] #Get preprocessed labels
    X_men = X[data.protect_list]
    y_men = y[data.protect_list]
    X_women = X[~data.protect_list]
    y_women = y[~data.protect_list]
    #data_men = torch.cat([X_men, y_men], axis=1)
    #data_women = torch.cat([X_women, y_women], axis=1)
    pd.DataFrame(X_men.numpy(), columns=list(range(X_men.shape[1]))).to_csv(f"{config['Basic']['dataset']}_inputs_men.csv")
    pd.DataFrame(X_women.numpy(), columns=list(range(X_women.shape[1]))).to_csv(f"{config['Basic']['dataset']}_inputs_women.csv")
    assert 1 == 0
    # If there is no learned model, learn a logistic regression
    # over the data and treat the regression as the known model
    if classifier_config["classifier"] == "Logistic_Regression":
        classifier = LogisticRegression()
        if y.shape[1] > 1:
            # If one-hot encoding labels, pass labels as the argmax
            classifier.create_model( X.detach(), torch.argmax( y, axis=1 ) )
        else:
            classifier.create_model( X.detach(), y.ravel() ) 
    elif classifier_config["classifier"] == "Sklearn_NN":
        classifier = Sklearn_NN( num_hidden=2 )
        if y.shape[1] > 1:
            # If one-hot encoding labels, pass labels as the argmax
            classifier.create_model( X.detach(), torch.argmax( y, axis=1 ) )
        else:
            classifier.create_model( X.detach(), y.ravel() ) 
    else:
        classifier = torch.load( classifier_config['model_path'] )
    
    # If we are using a variational autoencoder in order to use a 
    # gaussian latent-space, read this model in. If no vae, then
    # set encoder and decoder to the identity.
    # data.normalize sets up differentiable pre-processing
    vae = None
    if os.path.exists( vae_config['model_path'] ):
        vae = torch.load( vae_config['model_path'] )
        for param in vae.parameters():
            param.requires_grad_( False )

        vae.eval()
        encoder = vae.encoder
        decoder = lambda x: vae.decoder( data.normalize( x )[0] )
    else:
        print( "Path: {} does not exist".format( vae_config['model_path'] ) )
        encoder = torch.nn.Identity()
        decoder = lambda x: data.normalize( x )[0]

    # Read in classifier and ignore its gradients
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_( False )
    rep = lambda x: classifier( x, rep=True )

    X_ = encoder( X ) # Pass the data through the encoder
    #_y_ = classifier( X )
    #print( torch.sum( _y_.argmax( axis=1 ) == y.argmax( axis=1 ) ) / 2000 )
    ################# SET REFERENCE HERE ###########################
    # Get the reference points
    if basic_config['dataset'] == 'Adult':
        idx_ = np.where( data.frame[ "native-country" ] == " Vietnam" )[0][3]
    else:
        idx_ = 1

    #ids_german = [ "1", "29", "35", "37", "44", "54", "56", "59", "62", "63", "68" ]
    #ids_lucas = [ "1", "2", "4", "5", "6", "9", "10", "11", "12", "14", "35", "37", "38", 
    #              "41", "42", "43", "44", "45", "46", "47",  "48" , "49", "50", "52", "53",
    #              "56",  "57", "58" ]
    #ids_adult = ["0", "3", "8", "12", "13", "623", "804", "3521",  ]
    #idx_ = 59

    #ids_adult = ['0', '1', '1007', '1010', '1031', '12', '13', '1393', '141', '1475', '1488', '15', '1527', '1568', '16', '1614', '1647', '1691', '17', '1817', '1862', '1929', '2', '20', '2016', '2031', '2049', '2079', '2086', '2088', '21', '2103', '2107', '2152', '2156', '2157', '2168', '2180', '2182', '2183', '22', '2216', '2218', '2236', '2250', '23', '2301', '2350', '2353', '2371', '2375', '2398', '2431', '2433', '2437', '2445', '2446', '2451', '25', '2583', '26', '2625', '2640', '2654', '2661', '2662', '2686', '27', '2745', '2778', '28', '29', '2941', '3', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '402', '41', '43', '44', '45', '46', '5', '50', '54', '588', '604', '74', '931', '949', '969', '992']
    ids_german = ['1', '10', '105', '106', '11', '113', '116', '118', '120', '124', '127', '129', '13', '131', '137', '143', '15', '155', '166', '169', '170', '172', '174', '175', '18', '180', '182', '184', '186', '188', '190', '191', '192', '194', '195', '197', '199', '203', '212', '213', '226', '227', '228', '230', '235', '236', '237', '240', '242', '249', '252', '265', '268', '273', '274', '278', '289', '29', '291', '295', '301', '302', '304', '307', '308', '313', '315', '331', '332', '333', '334', '335', '337', '349', '35', '351', '353', '355', '357', '359', '364', '368', '37', '375', '378', '4', '44', '54', '56', '59', '62', '63', '68', '74', '76', '80', '87', '89', '92', '95']
    ids = [ int( idx ) for idx in ids_german ]
    graph = None
    """
    for idx_, val in enumerate( y.argmax( axis=1 ) ):
    
        if ( val == 1 ):
            continue

        """
    for idx_ in ids:
        
        print( "Idx: ", idx_ )
        print( y[idx_] )
        ###############################################################

        # Create reference data and read in desired output
        reference = ( X_[idx_], y[idx_], idx_ )
        print( "-----------------")
        print( reference[1] )
        desired_output = eval( classifier_config['desired_output'] )
        desired_output = torch.tensor( desired_output, dtype=X.dtype )

        #Setup Loss Fn
        if classifier_config['loss'] == "BCE":
            loss_fn = torch.nn.BCELoss( reduction='sum' )
        elif classifier_config['loss'] == "MSE":
            loss_fn = torch.nn.MSELoss( reduction='sum' )
        elif classifier_config['loss'] == "CrossEntropy":
            loss_fn = torch.nn.CrossEntropyLoss( reduction='sum' )

        # Get counterfactuals
        counterfactuals = None

        immutable_labels = data.where_label( eval( data_config['immutable_columns'] ) )
        immutable_idxs = []
        for idx in immutable_labels.values():
            immutable_idxs += idx
            
        non_decreasing_labels = data.where_label( eval( data_config['non_decreasing_columns'] ) )
        non_decreasing_idxs = []
        for idx in non_decreasing_labels.values():
            non_decreasing_idxs += idx

        non_increasing_labels = data.where_label( eval( data_config['non_increasing_columns'] )  )
        non_increasing_idxs = []
        for idx in non_increasing_labels.values():
            non_increasing_idxs += idx

        if basic_config['explainer'] == "DiCE":
            # If using Dice, Normalize by median absolute deviation
            mad = torch.median( torch.median( X_.squeeze(), axis=0 ).values )
            if mad == 0:
                mad = 1

            counterfactuals = dice.optimise_counterfactuals(
                    reference,
                    desired_output,
                    classifier, 
                    loss_fn,
                    decoder = decoder,
                    num_counterfactuals=int( basic_config["num_counterfactuals"] ), 
                    cov = None,
                    gaussian_reg = Gaussian( data.mean(), data.covariance() ),
                    mad = mad,
                    lr = float( dice_config["lr"] ),
                    iters = int( dice_config["iters"] ),
                    prox_reg = float( dice_config["prox_reg"] ),
                    dpp_reg = float( dice_config["dpp_reg"] ),
                    immutable_idxs = immutable_idxs,
                    non_decreasing_idxs = non_decreasing_idxs,
                    non_increasing_idxs = non_increasing_idxs
            )
        elif basic_config['explainer'] == 'GradualConstruction':
            counterfactuals = gradual_construction.optimise_counterfactuals(
                    reference,
                    desired_output,
                    classifier, 
                    loss_fn,
                    decoder = decoder,
                    num_counterfactuals=int( basic_config["num_counterfactuals"] ), 
                    acc = float( grad_con_config["accuracy"] ),
                    lr = float( grad_con_config["lr"] ),
                    iters = int( grad_con_config["iters"] ),
                    prox_reg = float( grad_con_config["prox_reg"] ),
                    dpp_reg = float( grad_con_config["dpp_reg"] ),
                    immutable_idxs = immutable_idxs,
                    non_decreasing_idxs = non_decreasing_idxs,
                    non_increasing_idxs = non_increasing_idxs
            )  
        elif basic_config['explainer'] == 'FACE':
            d = torch.ones(1)
            n_neighbors = int( face_config['n_neighbors'] )
            if face_config['method'] == "KNN":
                vol = torch.pi**(d/2) / torch.exp( torch.lgamma( d/2 + 1 ) )
                r = n_neighbors / ( X.shape[0] * vol )
                weight_function = lambda x: - x * np.log( r / x )

            counterfactuals, graph = face_knn.get_counterfactuals( 
                X_, 
                reference, 
                desired_output, 
                classifier,
                decoder = decoder,
                num_counterfactuals=int( basic_config["num_counterfactuals"] ),
                distance_threshold = float( face_config["distance_threshold"] ), 
                n_neighbors = int( face_config["n_neighbors"] ),
                weight_function = weight_function,
                immutable_idxs = immutable_idxs,
                non_decreasing_idxs = non_decreasing_idxs,
                non_increasing_idxs = non_increasing_idxs,
                graph = graph
            )

            if counterfactuals.shape[0] == 0:
                continue

        elif 'Bayes' in basic_config['explainer']:
            # If using bayes method, read in which columns need feasibility constraints
            immutable_columns = eval( data_config['immutable_columns'] )
            nonactionable_columns = eval( data_config[ 'nonactionable_columns' ] )

            if vae is not None:
                # If we're using a vae, latent space is always standard gaussian
                mu = torch.zeros( X_.shape[1] )
                cov = torch.eye( X_.shape[1] )
            else:
                # Get mean and covariance from data distribution
                mu = data.mean()
                cov = data.covariance()

            # Initialize CF Distribution
            prior_x = CounterfactualDistribution( mu, cov )
            prior_x.set_rep( rep )
            prior_x.set_dec( decoder )
            prior_x.set_enc( encoder )
            prior_x.set_data( X_, y )
            prior_x.set_desired_label( desired_output )

            #Perform Laplace Approximation
            prior_x_w_given_y = prior_x.laplace_apx( classifier,
                    laplace_iters = int( bayes_config['laplace_iters'] ),
                    x_lr = float( bayes_config['x_lr'] ),
                    x_iters = int( bayes_config['x_iters'] ),
            )

            #Marginalize over weights
            w_idxs_len = prior_x_w_given_y.loc.shape[0] - prior_x.loc.shape[0]
            prior_x_given_y = prior_x_w_given_y.marginalize_over( range( w_idxs_len ) )

            # Setup prior over joint x and x' distribution
            alpha = float( bayes_config['alpha'] )
            joint_mu = torch.hstack( [ mu, mu ] )

            if len( immutable_columns ) > 0:
                # If anything is immutable adjust covariance accordingly
                feasible_cov = data.covariance( feasible=True )
                cov_ = torch.logical_not( feasible_cov.to( bool ) ) * cov
                feasible_cov = alpha * feasible_cov + cov_
            else:
                feasible_cov = alpha * cov

            joint_cov = torch.vstack( [
                    torch.hstack( [ cov, feasible_cov ] ),
                    torch.hstack( [ feasible_cov, cov ] )
                ] )
            
            # Joint prior over x and x'
            joint_prior = Gaussian( joint_mu, joint_cov, )

            # Get product p( x, x' | y ) = p( x, x'  ) p( x' | y )
            joint_cf = cf_posterior( joint_prior, prior_x_given_y, torch.arange( prior_x.loc.shape[0] ) )

            # Condition on reference p( x' | x, y )
            cf_dist = joint_cf.conditioned_on( reference[0].squeeze(), torch.arange( prior_x.loc.shape[0] ) )

            if len( nonactionable_columns ) > 0:
                # If any non-actionable columns, update mean and cov accordingly
                feasible_mu, feasible_cov = data.nonactionable_params( 
                    cf_dist.mean, 
                    cf_dist.covariance_matrix )

                cf_dist = Gaussian( feasible_mu, feasible_cov )
            
            # Read in number of counterfacuals, sample from the distribution,
            # andn decode the back into the correct format
            if basic_config['explainer'] == 'Bayes':
                num_counterfactuals = int( basic_config['num_counterfactuals'] )
                counterfactuals_raw = cf_dist.sample([num_counterfactuals] )
                counterfactuals = decoder( counterfactuals_raw )
            elif basic_config['explainer'] == 'Bayes-Face':
                d = torch.ones(1)
                n_neighbors = int( face_config['n_neighbors'] )
                if face_config['method'] == "KNN":
                    vol = torch.pi**(d/2) / torch.exp( torch.lgamma( d/2 + 1 ) )
                    r = n_neighbors / ( X.shape[0] * vol )
                    weight_function = lambda x: - x * np.log( r / x )

                counterfactuals = face_knn.get_counterfactuals( 
                    X_, 
                    reference, 
                    desired_output, 
                    classifier,
                    decoder = decoder,
                    num_counterfactuals=int( basic_config["num_counterfactuals"] ),
                    distance_threshold = float( face_config["distance_threshold"] ), 
                    n_neighbors = int( face_config["n_neighbors"] ),
                    cov = cf_dist.covariance_matrix,
                    weight_function = weight_function,
                    immutable_idxs = immutable_idxs,
                    non_decreasing_idxs = non_decreasing_idxs,
                    non_increasing_idxs = non_increasing_idxs
                )
            elif basic_config['explainer'] == 'Bayes-DiCE':
                counterfactuals = dice.optimise_counterfactuals(
                        reference,
                        desired_output,
                        classifier, 
                        loss_fn,
                        decoder = decoder,
                        num_counterfactuals=int( basic_config["num_counterfactuals"] ), 
                        cov = cf_dist.covariance_matrix,
                        lr = float( dice_config["lr"] ),
                        iters = int( dice_config["iters"] ),
                        prox_reg = float( dice_config["prox_reg"] ),
                        dpp_reg = float( dice_config["dpp_reg"] ),
                        immutable_idxs = immutable_idxs,
                        non_decreasing_idxs = non_decreasing_idxs,
                        non_increasing_idxs = non_increasing_idxs
                )
        # Write counterfactuals out to file
        # TODO: Convert counterfactuals, and labels to DataFrame and write DF as CSV
        filename = ".{}.{}".format( idx_, basic_config["explainer"].lower() )
        out_file = basic_config["out_folder"] + filename

        # Get predictions
        if basic_config["dataset"].lower() == "mnist":
            # If we are using mnist, decoding requires reshaping
            cf_decode, _ = data.decode( counterfactuals.detach().numpy().squeeze() )
            cf_decode = cf_decode.reshape( -1, 28, 28 )
            cf_encode, _ = data.encode( cf_decode )
            c_predictions = classifier( torch.tensor( cf_encode ) ).detach()

            ref_out = classifier( decoder( reference[0] ) ).detach().numpy()
            ref_x, _ = data.decode( decoder( reference[0] ).unsqueeze(0).numpy() )
            ref_out = np.hstack( [
                ref_x.flatten(),
                ref_out.flatten() ] )
            ref_out = ref_out.reshape( 1, -1 )
        else:
            c_predictions = classifier( counterfactuals ).detach()

            cf_decode, _ = data.decode( counterfactuals )
            cf_encode, _ = data.encode( np.hstack( [ cf_decode, np.zeros( ( cf_decode.shape[0], 1 ) ) ] ) )      
            c_predictions = classifier( cf_encode ).detach()

            print( "Average Label: ", torch.mean( c_predictions , axis=0 ) )

            c_predictions = data.features[ data_config['target_column'] ].decode( c_predictions )
            
            target_decode = data.features[ data_config['target_column'] ].decode
            ref_out = classifier( reference[0] ).detach().numpy()

            ref_x, _ = data.decode( reference[0].unsqueeze(0).numpy() )
            ref_out = np.hstack( [
                ref_x.flatten(),
                target_decode( classifier( reference[0] ).detach().numpy() ).flatten() ] )
            ref_out = ref_out.reshape( 1, -1 )

        # Gather counterfactuals for output
        out = np.hstack( [
            np.vstack( [ el.flatten() for el in cf_decode ] ),
            c_predictions ] )
            
        desired_label = data.features[ data.target_column ].decode( desired_output.reshape(1,-1) )
        desired_label = desired_label.squeeze()
        
        idxs = np.where( c_predictions == desired_label )[0]
        out = out[ idxs, : ]

        df_ref = pd.DataFrame( ref_out, columns=data.columns )
        df = pd.DataFrame( out, columns=data.columns )
        
        #Output explanations with format:
        #   ref
        #   {empty line}
        #   cf_1
        #   ...
        #   cf_n

        print_df = df_ref.append( pd.Series( dtype=np.float32 ), ignore_index=True )
        print_df = print_df.append( df, ignore_index=True )
        print_df.to_csv( out_file, sep=",", index=False )

