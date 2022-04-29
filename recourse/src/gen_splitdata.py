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
    X_women = X[data.protect_list]
    y_women = y[data.protect_list]
    X_men = X[~data.protect_list]
    y_men = y[~data.protect_list]
    #data_men = torch.cat([X_men, y_men], axis=1)
    #data_women = torch.cat([X_women, y_women], axis=1)
    pd.DataFrame(X_men.numpy(), columns=list(range(X_men.shape[1]))).to_csv(f"{config['Basic']['dataset']}_inputs_notprotec.csv")
    pd.DataFrame(X_women.numpy(), columns=list(range(X_women.shape[1]))).to_csv(f"{config['Basic']['dataset']}_inputs_protec.csv")
    
