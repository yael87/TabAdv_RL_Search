
import sys
import pandas as pd
import numpy as np
import configparser
import random
import pickle
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#from Models.scikitlearn_wrapper import SklearnClassifier

# prepare data and models
from Utils.data_utils import preprocess_credit, drop_corolated_target, split_to_datasets, preprocess_top100, preprocess_HCDR, \
                            preprocess_ICU, preprocess_HATE,rearrange_columns, rearrange_columns_edittable, over_sampling, \
                            write_edditable_file

from Utils.models_utils import load_target_models, load_surrogate_model, train_GB_model, train_LGB_model, train_RF_model, train_XGB_model,  \
                            train_REGRESSOR_model, compute_importance

from Utils.surrogate_model import train_SURR_NN_model

#scalers
from Classes.TorchMinMaxScaler import TorchMinMaxScaler

# wrappers
from art.estimators.classification import XGBoostClassifier, SklearnClassifier
from Models.lightgbm_wrapper import LightGBMClassifier

# attacks
from art.attacks.evasion import HopSkipJump, BoundaryAttack
from Attacks.Boundary_Orig import BoundaryAttack_
from Attacks.Boundary_Orig_Scaled import BoundaryAttack_s
from Attacks.HopSkipJump_Orig import HopSkipJump_
from Utils.attack_utils import get_hopskipjump, get_constraints, get_surrogate_Attack, get_boundary, \
                                get_balanced_attack_set, get_attack_set, get_adv_init, load_attacks
from Utils.generate_attacks import  surrogate_attack, query_based_attack


def get_config():
    config = configparser.ConfigParser()
    #config.read(sys.argv[1])
    config.read('configurations.txt')
    config = config['DEFAULT']
    return config




if __name__ == '__main__':

    # Set parameters
    
    configurations = get_config()
    data_path = configurations["data_path"]
    raw_data_path = configurations["raw_data_path"]
    perturbability_path = configurations["perturbability_path"]
    results_path = configurations["results_path"]
    seed = int(configurations["seed"])
    dataset_name = raw_data_path.split("/")[1]
    models_path = configurations['models_path']
    exp_type = configurations['exp_type']
    
    # Get dataset
    datasets = split_to_datasets(raw_data_path, save_path=data_path)
    
    # Get scalers
    scaler = pickle.load(open(data_path+"/scaler.pkl", 'rb' ))
    scaler_pt = pickle.load(open(data_path+"/scaler_pt.pkl", 'rb' ))
    constraints, perturbability = get_constraints(dataset_name, perturbability_path)

    columns_names = list(datasets.get('x_test').columns)

    #process(dataset_name, raw_data_path, TorchMinMaxScaler)
    #train_models(data_path, datasets)
    #train_REG_models(dataset_name, data_path, datasets) 

    
    # Get models
    GB, LGB, XGB, RF = load_target_models(data_path ,models_path)
    SURR = load_surrogate_model(data_path ,models_path)

    target_models = [XGB]#, XGB, LGB, RF]
    surr_model = SURR
        
    target_models_names = ["XGB"]#, "XGB", "LGB", "RF"]
    surr_model_names = "SURR"


    


