import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample, shuffle
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from Utils.surrogate_model import train_SURR_NN_model

#from pyimagesearch import mlp
from collections import OrderedDict
from torch.optim import Adam
import torch.nn as nn
import torch
import wandb 


def train_models(data_path, datasets):
    # Train models
    GB, gb_eval = train_GB_model(data_path, datasets=datasets, model_type="target")
    LGB, lgb_eval = train_LGB_model(data_path, datasets=datasets, model_type="target")
    RF, rf_eval = train_RF_model(data_path, datasets=datasets, model_type="target")
    XGB, rf_eval = train_XGB_model(data_path, datasets=datasets, model_type="target")
    #SURR, surr_eval = train_SURR_NN_model(data_path, datasets=datasets, model_type="surrogate")
    
def train_REG_models(dataset_name, data_path, datasets):
    if 'ICU' in dataset_name:
        REG_apache_3j_bodysystem, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="apache_3j_bodysystem")
        REG_apache_3j_diagnosis, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="apache_3j_diagnosis") 
        REG_d1_mbp_invasive_max, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="d1_mbp_invasive_max")
        REG_d1_mbp_invasive_min, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="d1_mbp_invasive_min")
        #REG_apache_3j_bodysystem = pickle.load(open(models_path+"/ICU_apache_3j_bodysystem_REG_seed-42_lr-0.1_estimators-200_maxdepth-6.pkl", 'rb' ))
        #REG_apache_3j_diagnosis = pickle.load(open(models_path+"/ICU_apache_3j_diagnosis_REG_seed-42_lr-0.1_estimators-200_maxdepth-6.pkl",'rb'))   

    if 'HCDR' in dataset_name:
        REG_DEF_30_CNT_SOCIAL_CIRCLE, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="DEF_30_CNT_SOCIAL_CIRCLE")
        REG_DEF_60_CNT_SOCIAL_CIRCLE, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="DEF_60_CNT_SOCIAL_CIRCLE")
        EXT_SOURCE_1, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="EXT_SOURCE_1")
        EXT_SOURCE_2, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="EXT_SOURCE_2") 
        EXT_SOURCE_3, reg_eval = train_REGRESSOR_model(data_path, datasets=datasets, model_type="EXT_SOURCE_3") 


def load_target_models(data_path, models_path):
    data_name = data_path.split("/")[-1]
    if "HATE" in data_name:
        GB = {'seed':42, 'lr': 2.5, 'estimators':40, 'maxdepth':7}
        LGB = {'seed':42, 'lr': 1.0, 'estimators':300, 'maxdepth':7}
        XGB = {'seed':42, 'lr': 1.0, 'estimators':90, 'maxdepth':3}
        RF = {'seed':42, 'estimators':100, 'maxdepth':4}
    if "HCDR" in data_name:
        GB = {'seed':42, 'lr': 1., 'estimators':500, 'maxdepth':5}
        LGB = {'seed':42, 'lr': 0.1, 'estimators':200, 'maxdepth':8}
        XGB = {'seed':42, 'lr': 2., 'estimators':300, 'maxdepth':7}
        RF = {'seed':42, 'estimators':500, 'maxdepth':9}
    if "ICU" in data_name:
        GB = {'seed':42, 'lr': 0.01, 'estimators':500, 'maxdepth':6}
        LGB = {'seed':42, 'lr': 0.1, 'estimators':200, 'maxdepth':8}
        XGB = {'seed':42, 'lr': 0.1, 'estimators':70, 'maxdepth':8}
        RF = {'seed':42, 'estimators':500, 'maxdepth':9}
    if "RADCOM" in data_name:
        GB = {'seed':42, 'lr': 1.0, 'estimators':300, 'maxdepth':5}
        LGB = {'seed':42, 'lr': 0.1, 'estimators':200, 'maxdepth':8}
        XGB = {'seed':42, 'lr': 1.0, 'estimators':300, 'maxdepth':5}
        RF = {'seed':42, 'estimators':500, 'maxdepth':9}

    model_name_LGB = "{}_{}_LGB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, 'target', LGB['seed'], LGB['lr'], LGB['estimators'], LGB['maxdepth'])
    model_name_GB = "{}_{}_GB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, 'target',GB['seed'], GB['lr'], GB['estimators'], GB['maxdepth'])
    model_name_RF = "{}_{}_RF_seed-{}_estimators-{}_maxdepth-{}".format(data_name, 'target', RF['seed'], RF['estimators'], RF['maxdepth'])
    model_name_XGB = "{}_{}_XGB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, 'target',XGB['seed'], XGB['lr'], XGB['estimators'], XGB['maxdepth'])
    #model_name_NN = "{}_{}_NN_seed-{}_lr-{}_batch_size-{}_epochs-{}".format(data_name, 'surrogate',seed, 0.01, 1024, 70)

    LGB  = pickle.load(open(models_path + "/" + model_name_LGB + ".pkl", 'rb'))
    GB  = pickle.load(open(models_path + "/" + model_name_GB + ".pkl", 'rb'))
    RF  = pickle.load(open(models_path + "/" + model_name_RF + ".pkl", 'rb'))
    XGB  = pickle.load(open(models_path + "/" + model_name_XGB + ".pkl", 'rb'))
    #NN = torch.jit.load(open(models_path + "/" + model_name_NN + ".pt", 'rb' ))

    return GB, LGB, XGB, RF

def load_surrogate_model(data_path, models_path):
    data_name = data_path.split("/")[-1]
    if "HATE" in data_name:
        SURR = {'seed':42, 'lr': 0.2, 'batch_size':1024, 'epochs':70}
    if "HCDR" in data_name:
        SURR = {'seed':42, 'lr': 0.042, 'batch_size':2048, 'epochs':14}
    if "ICU" in data_name:
        SURR = {'seed':42, 'lr': 0.01, 'batch_size':1024, 'epochs':70}
    if "RADCOM" in data_name:
        SURR = {'seed':42, 'lr': 0.2, 'batch_size':1024, 'epochs':15}
   
    model_name_NN = "{}_{}_NN_seed-{}_lr-{}_batch_size-{}_epochs-{}".format(data_name, 'surrogate',SURR['seed'], SURR['lr'], SURR['batch_size'], SURR['epochs'])
    
    SURR = torch.jit.load(open(models_path + "/" + model_name_NN + ".pt", 'rb' ))
    #SURR = pickle.load(open(models_path + "/" + model_name_NN + ".pkl", 'rb' ))
    #print(SURR)
    SURR.eval()

    return SURR
'''
def train_SURR_NN_model(data_path, seed=42, val_size=0.2, learning_rate=0.001, BATCH_SIZE=64, EPOCHS =100,
                   saving_path="Models/", datasets=None, model_type="surrogate", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    scaler = pickle.load(open(data_path+"/scaler.pkl", 'rb' ))
    data_name = data_path.split("/")[-1]
 
    #x_train = torch.tensor(x_train.values, dtype=torch.float32)
    #y_train = torch.tensor(y_train, dtype=torch.float32)
    #x_val = torch.tensor(x_val.values, dtype=torch.float32)
    #y_val = torch.tensor(y_val, dtype=torch.float32)
    x_train = torch.from_numpy(scaler.transform(x_train)).float()
    y_train = torch.from_numpy(y_train.to_numpy()).float()
    x_val = torch.from_numpy(scaler.transform(x_val)).float()
    y_val = torch.from_numpy(y_val.to_numpy()).float()


    def get_training_model(inFeatures=x_train.shape[1], hiddenDim=256, nbClasses=1):
        # construct a shallow, sequential neural network
        mlpModel = nn.Sequential(OrderedDict([
            ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
            ("activation_1", nn.ReLU()),
            ("batch_norm_1", nn.BatchNorm1d(hiddenDim)),
            ("hidden_layer_2", nn.Linear(hiddenDim, 16)),
            ("activation_2", nn.ReLU()),
            ("batch_norm_2", nn.BatchNorm1d(16)),
            ("hidden_layer_3", nn.Linear(16, 8)),
            ("activation_3", nn.ReLU()),
            ("batch_norm_3", nn.BatchNorm1d(8)),
            ("output_layer", nn.Linear(8, nbClasses)),
            ("sigmoid_layer",  nn.Sigmoid())
        ]))
        # return the sequential model
        return mlpModel
    
    def next_batch(inputs, targets, batchSize):
        # loop over the dataset
        for i in range(0, inputs.shape[0], batchSize):
            # yield a tuple of the current batched data and labels
            yield (inputs[i:i + batchSize], targets[i:i + batchSize])
    
    # determine the device we will be using for training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] training using {}...".format(DEVICE))

    # initialize our model and display its architecture
    SURR = get_training_model().to(DEVICE)
    print(SURR)
    # initialize optimizer and loss function
    opt = Adam(SURR.parameters(), lr=LR)
    lossFunc = nn.BCELoss()
    
    # create a template to summarize current training progress
    trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
    # loop through the epochs
    for epoch in range(0, EPOCHS):
        # initialize tracker variables and set our model to trainable
        print("[INFO] epoch: {}...".format(epoch + 1))
        trainLoss = 0
        trainAcc = 0
        samples = 0
        SURR.train()
        # loop over the current batch of data
        for (batchX, batchY) in next_batch(x_train, y_train, BATCH_SIZE):
            # flash data to the current device, run it through our
            # model, and calculate loss
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
            opt.zero_grad()
            predictions = SURR(batchX)
            loss = lossFunc(predictions, batchY.float())
            predictions = torch.where(SURR(batchX) > 0.5, 1.0, 0.0)
            # zero the gradients accumulated from the previous steps,
            # perform backpropagation, and update model parameters
            #opt.zero_grad()
            loss.backward()
            opt.step()
            # update training loss, accuracy, and the number of samples
            # visited
            trainLoss += loss.item() * batchY.size(0)
            a = (predictions == batchY).sum().item()
            b = (predictions == batchY).sum()
            trainAcc += (predictions == batchY).sum().item()
            samples += batchY.size(0)
        # display model progress on the current training batch
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples),
            (trainAcc / samples)))
    
        # initialize tracker variables for testing, then set our model to
        # evaluation mode
        testLoss = 0
        testAcc = 0
        samples = 0
        SURR.eval()
        # initialize a no-gradient context
        with torch.no_grad():
            # loop over the current batch of test data
            for (batchX, batchY) in next_batch(x_val, y_val, BATCH_SIZE):
                # flash the data to the current device
                (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
                # run data through our model and calculate loss
                predictions = SURR(batchX)
                loss = lossFunc(predictions, batchY.float())
                predictions = torch.where(SURR(batchX) > 0.5, 1.0, 0.0)
                # update test loss, accuracy, and the number of
                # samples visited
                testLoss += loss.item() * batchY.size(0)
                testAcc += (predictions == batchY).sum().item()
                samples += batchY.size(0)
            # display model progress on the current test batch
            testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
            print(testTemplate.format(epoch + 1, (testLoss / samples),
                (testAcc / samples)))
            print("")
    
    ########
    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/" 
    
    model_name = "{}_{}_NN_seed-{}_lr-{}_BATCH_SIZE-{}_EPOCHS-{}".format(data_name, model_type, seed, learning_rate,
                                                                                          BATCH_SIZE, EPOCHS)
    pickle.dump(SURR, open(saving_path + model_name + ".pkl", 'wb'))                                                                                     
    
    model_name=model_name+'_upsamptest'
    eval = model_evaluation(model=SURR.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return SURR, eval
'''

def train_REGRESSOR_model(data_path, seed=42, val_size=0.2, learning_rate=0.01, n_estimators=200, max_depth=6,
                   saving_path="Models/", datasets=None, model_type="surrogate", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_surrogate"]
        x_val = datasets["x_test"]
        c = datasets["y_train_surrogate"]
        y_val = datasets["y_test"]
    
    y_train = x_train[model_type]
    x_train = x_train.drop([model_type], axis=1)
    y_val = x_val[model_type]
    x_val = x_val.drop([model_type], axis=1)
    
    REG = GradientBoostingRegressor(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=seed,)
                                    #loss='absolute_error')


    REG.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/" 
    if exclude is None:
        model_name = "{}_{}_REG_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate,
                                                                                          n_estimators, max_depth)
    else:
        model_name = "{}_{}_REG_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,
                                                                                          exclude, seed, learning_rate,
                                                                                           n_estimators, max_depth)
    pickle.dump(REG, open(saving_path + model_name + ".pkl", 'wb'))  
    
    eval = regression_evaluation(model=REG.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)
    
    

    return REG, eval


def train_GB_model(data_path, seed=42, val_size=0.2, learning_rate=0.1, n_estimators=200, max_depth=5,
                   saving_path="Models/", datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    GB = GradientBoostingClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=seed)

   
    GB.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/" 
    if exclude is None:
        model_name = "{}_{}_GB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate,
                                                                                          n_estimators, max_depth)
    else:
        model_name = "{}_{}_GB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,
                                                                                          exclude, seed, learning_rate,
                                                                                           n_estimators, max_depth)
    pickle.dump(GB, open(saving_path + model_name + ".pkl", 'wb'))                                                                                     
    
    model_name=model_name+'_upsamptest'
    eval = model_evaluation(model=GB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return GB, eval

def train_LGB_model(data_path, seed=42, val_size=0.2, learning_rate=0.1, n_estimators=400, max_depth=8,
                   saving_path="Models/", datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]
    
    LGB = LGBMClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=seed)
       
    
    LGB.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/" 
    if exclude is None:
        model_name = "{}_{}_LGB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate,
                                                                                          n_estimators, max_depth)
    else:
        model_name = "{}_{}_LGB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,
                                                                                          exclude, seed, learning_rate,
                                                                                        n_estimators, max_depth)
    pickle.dump(LGB, open(saving_path + model_name + ".pkl", 'wb'))                                                                                     
    
    model_name=model_name+'_upsamptest'
    eval = model_evaluation(model=LGB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return LGB, eval


def train_RF_model(data_path, seed=42, val_size=0.2, n_estimators=400, max_depth=7, saving_path="Models/",
                   datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    RF = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=seed)
    RF.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/"

    if exclude is None:
        model_name = "{}_{}_RF_seed-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, n_estimators, max_depth)
    else:
        model_name = "{}_{}_RF_exclude_{}_seed-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,  exclude, seed, n_estimators, max_depth)

    pickle.dump(RF, open(saving_path + model_name + ".pkl", 'wb'))

    model_name=model_name+'_upsamptest'
    eval = model_evaluation(model=RF.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return RF, eval

def train_XGB_model(data_path, seed=42, val_size=0.2, learning_rate=0.1, n_estimators=300,  max_depth=7, saving_path="Models/",
                   datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    estimator = XGBClassifier(objective= 'binary:hinge', nthread=4, seed=seed,
                                                                    n_estimators = n_estimators,
                                                                    learning_rate = learning_rate,
                                                                    max_depth = max_depth, 
                                                                    use_label_encoder=False)
    #parameters = {'max_depth': range (1, 10, 1), 'n_estimators': range(5, 100, 5), 'learning_rate': [0.5, 0.1, 0.01, 0.05]}
    '''
    if( "HATE" in dataset_file):
        param_test1 = {'n_estimators': range(20, 100, 10), 'max_depth': range(1, 5, 1)}#10
    else:
        param_test1 = {'n_estimators': range(20, 1000, 10), 'max_depth': range(1, 5, 1)}#30
    '''
    '''
    gsearch = GridSearchCV(
        estimator = estimator, param_grid=parameters, scoring='accuracy',  n_jobs=-1,  cv=5)
    
    #gsearch = GridSearchCV(
    #    estimator=HistGradientBoostingClassifier(learning_rate=0.1, random_state=1234),
    #    param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    #print("XGB: performing hyperparameter tuning using GridSearchCV, cv=5...")
    gsearch.fit(x_train.to_numpy(), y_train)

    print("Best params found: " + str(gsearch.best_params_)) #0.1 70 8
    gsearch.best_estimator_.fit(x_train, y_train)
    XGB = gsearch.best_estimator_
    '''
    XGB = estimator.fit(x_train.to_numpy(), y_train)
    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/"

    if exclude is None:
        model_name = "{}_{}_XGB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate, n_estimators, max_depth)
    else:
        model_name = "{}_{}_XGB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,  exclude, seed, learning_rate, n_estimators, max_depth)

    pickle.dump(XGB, open(saving_path + model_name + ".pkl", 'wb'))

    model_name=model_name+'_upsamptest'

    eval = model_evaluation(model=XGB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return XGB, eval


def model_evaluation(model, val_x, val_y, saving_path="Models/Evaluation/", model_name="model"):
    if '_1' in model_name:
        pos_label = 1
    elif '_0' in model_name:
        pos_label = 0
    else:
        pos_label = 1
   
    eval = {
        "accuracy_score": [accuracy_score(val_y, model(val_x))],
        "f1_score": [f1_score(val_y, model(val_x), pos_label=pos_label)],
        "precision_score": [precision_score(val_y, model(val_x))],
        "recall_score": [recall_score(val_y, model(val_x))]
    }
    pd.DataFrame(eval).to_csv(saving_path + model_name + ".csv", index=False)
    log_eval = {x:eval[x][0] for (x,_) in eval.items()}
    #wandb.log(log_eval)
    return eval

def model_NN_evaluation(model, val_x, val_y, saving_path="Models/", model_name="model", scaler=None):
    if (scaler):
        val_x = scaler.transform(val_x)

    preds = torch.round(nn.Sigmoid(model(val_x))).detach().numpy()
    eval = {
        "accuracy_score": [accuracy_score(val_y, preds)],
        "f1_score": [f1_score(val_y, preds, pos_label=0)],
        "precision_score": [precision_score(val_y, preds)],
        "recall_score": [recall_score(val_y, preds)]
    }
    pd.DataFrame(eval).to_csv(saving_path + "Evaluation/" + model_name + ".csv", index=False)
    return eval

def regression_evaluation(model, val_x, val_y, saving_path="Models/", model_name="model"):
    eval = {
        "mse": [mean_squared_error(val_y, model(val_x))],
        "rmse": [ math.sqrt(mean_squared_error(val_y, model(val_x)))],
        "mae": [mean_absolute_error(val_y, model(val_x))],
    }
    pd.DataFrame(eval).to_csv(saving_path + "Evaluation/" + model_name + ".csv", index=False)
    return eval

def compute_importance(target_models, target_models_names, data_path):

    for j, target in enumerate(target_models):
        
        model  = target
        importances = target.feature_importances_
        
        print(target_models_names[j]+':')
        imp = []
        for i,v in enumerate(importances):
            print('Feature: %0d, Score: %.5f' % (i,v))
            imp.append(v)
            # plot feature importance
        plt.bar([x for x in range(len(importances))], importances)
        plt.show()
        plt.close()
        imp = pd.DataFrame(imp)
        imp.to_csv(data_path+"/importance_{}.csv".format(target_models_names[j]), index=False)

# pickle.load(open(filename, 'rb'))

def load_AE_models(models_path):
    models = {}
    models['AE_0'] = torch.jit.load(open(models_path + "/autoencoder_0_model_epochs-9.pt", 'rb'))
    models['AE_0'].eval()
    models['AE_1'] = torch.jit.load(open(models_path + "/autoencoder_1_model_epochs-9.pt", 'rb'))
    models['AE_1'].eval()

    return models

def load_IF_models(models_path):
    models = {}
    models['IF_0'] = pickle.load(open(models_path + "/isolation_forest_0_model.pkl", 'rb'))
    models['IF_1'] = pickle.load(open(models_path + "/isolation_forest_1_model.pkl", 'rb'))

    return models