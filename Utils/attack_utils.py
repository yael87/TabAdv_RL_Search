import numpy as np
import pandas as pd
#from art.attacks.evasion import AdversarialPatchPyTorch
#from Attacks.HopSkipJump_Tabular import HopSkipJump
#from Attacks.Boundary_Tabular import BoundaryAttack

#from Attacks.Surrogate import SurrogateAttack
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import preprocessing
import pickle
import torch
from Classes.TorchMinMaxScaler import TorchMinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

class FactoryCat:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def method(self, vector):
        vector = integer(vector)
        return bound(vector, self.min, self.max)

    def get_method(self):
        return self.method

class Bound:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def method(self, vector):
        return bound(vector, self.min, self.max)

    def get_method(self):
        return self.method


def bound(vector, v_min=-np.inf, v_max=np.inf):
    v = np.clip(vector, v_min, v_max)
    return v

def normalized(vector, v_min=0, v_max=0):
    return bound(vector, 0.0, 0.067452258965999)

def normalized_negative(vector, v_min=0, v_max=0):
    return bound(vector, -1.0, 1.0)

def positive(v_min=0, v_max=np.inf):
    return Bound(v_min, v_max).get_method()

def positive_(v_min=0, v_max=np.inf):
    return TorchMinMaxScaler(feature_range=(v_min, v_max)).get_method()

def integer(vector, v_min=0, v_max=0):
    return np.round(vector)


def binary(vector, v_min=0, v_max=0):
    vector = integer(vector)
    return bound(vector, 0, 1)


def negative(v_min=-np.inf, v_max=0):
    return Bound(v_min, v_max).get_method()


def categorical(v_min=0, v_max=0):
    vec = FactoryCat(v_min, v_max).get_method()
    #vec = preprocessing.minmax_scale(vector.T, feature_range=(1,3)).T
    return vec

def cond(vector, cond=None, val=0):
    if (cond != None):
        if(eval(str(vector)+str(cond))):
            v = np.clip(vector, val, val)
        return v

def increas(gap, vector, pre_vector):
    return np.greater(np.min(vector, np.add(pre_vector,gap)), pre_vector)

def bmi(vector, col_names):
    weight_ind = col_names.index("weight")
    height_ind = col_names.index('height')
    try:
        vec = vector.data.squeeze()
    except:
        vec = vector.flatten()
    
    if (vec[weight_ind]> 0):
        bmi = vec[height_ind]/vec[weight_ind]
    else:
        bmi=0
    return bmi

def compute_reg_value(reg_model, vector, col_idx, col_names):
    try:
        vector = vector.clone().detach().squeeze()
        x_vec = torch.cat([vector[0:col_idx], vector[col_idx+1:]]) #drop col_idx feature
        reg = reg_model.predict(x_vec.numpy().reshape(1, -1))
        reg = torch.from_numpy(reg)
    except:
        vector = vector.copy()#.flatten()
        x_vec = np.delete(vector, col_idx, axis = 1) # remove the column at col_idx
        #x_vec = np.concatenate((vector[:][0:col_idx], vector[:][col_idx+1:])) #drop col_idx feature
        reg = reg_model.predict(x_vec)
    return reg

def DEF_30_CNT_SOCIAL_CIRCLE(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/HCDR/HCDR_DEF_30_CNT_SOCIAL_CIRCLE_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def DEF_60_CNT_SOCIAL_CIRCLE(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/HCDR/HCDR_DEF_60_CNT_SOCIAL_CIRCLE_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def EXT_SOURCE_1(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/HCDR/HCDR_EXT_SOURCE_1_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def EXT_SOURCE_2(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/HCDR/HCDR_EXT_SOURCE_2_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def EXT_SOURCE_3(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/HCDR/HCDR_EXT_SOURCE_3_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def apache_3j_bodysystem(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/ICU/ICU_apache_3j_bodysystem_REG_seed-42_lr-0.1_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def apache_3j_diagnosis(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/ICU/ICU_apache_3j_diagnosis_REG_seed-42_lr-0.1_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def d1_mbp_invasive_max(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/ICU/ICU_d1_mbp_invasive_max_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

def d1_mbp_invasive_min(vector, col_idx, col_names):
    reg_model = pickle.load(open("Models/ICU/ICU_d1_mbp_invasive_min_REG_seed-42_lr-0.01_estimators-200_maxdepth-6.pkl", 'rb' ))
    reg = compute_reg_value(reg_model, vector, col_idx, col_names)
    return reg

'''
def date_change(current):
    # year and month are not change. only the day
    dates = []
    new_date = current.copy()  # 20180200
    while (
            new_date / 100 == current / 100 and new_date % 100 <= 30):  # stay in same year and month, day can increase until 30
        new_date = new_date + 1
        dates.append(new_date)

    return dates


def time_change(current):
    new_time = current.copy()
    times = []
    new_date = current.copy()  # 235959

    while (new_time / 10000 < 24):
        while ((new_time / 100) % 100 < 60):
            while (new_time % 100 < 60):
                new_time = new_time + 29  # should be 1
                times.append(new_time)
            new_time = (new_time / 100 + 2) * 100  # add minute #should be +1
            times.append(new_time)
        new_time = (new_time / 10000 + 1) * 10000  # add hour
        times.append(new_time)

    return times
'''

def get_constraints(dataset_name, perturbability_path):
    perturbability = pd.read_csv(perturbability_path)
    perturbability = perturbability["Perturbability"].to_numpy()

    if dataset_name == "HATE":
        constraints = {
            'hate_neigh': [binary],
            'normal_neigh': [binary],
            'statuses_count':[positive(),integer],
            'favorites_count':[positive(),integer],
            'listed_count':[positive(),integer],
            'science_empath':[normalized],
            'horror_empath':[normalized],
            'sadness_empath':[normalized],
            'internet_empath':[normalized],
            'college_empath':[normalized],
            'attractive_empath':[normalized],
            'technology_empath':[normalized],
            'messaging_empath':[normalized],
            'surprise_empath':[normalized],
            'furniture_empath':[normalized],
            'restaurant_empath':[normalized],
            'domestic_work_empath':[normalized],
            'art_empath':[normalized],
            'sympathy_empath':[normalized],
            'anger_empath':[normalized],
            'neglect_empath':[normalized],
            'farming_empath':[normalized],
            'beauty_empath':[normalized],
            'white_collar_job_empath':[normalized],
            'fabric_empath':[normalized],
            'confusion_empath':[normalized],
            'contentment_empath':[normalized],
            'night_empath':[normalized],
            'ridicule_empath':[normalized],
            'health_empath':[normalized],
            'zest_empath':[normalized],
            'medical_emergency_empath':[normalized],
            'sports_empath':[normalized],
            'trust_empath':[normalized],
            'cleaning_empath':[normalized],
            'divine_empath':[normalized],
            'urban_empath':[normalized],
            'anonymity_empath':[normalized],
            'hipster_empath':[normalized],
            'ocean_empath':[normalized],
            'crime_empath':[normalized],
            'play_empath':[normalized],
            'home_empath':[normalized],
            'anticipation_empath':[normalized],
            'children_empath':[normalized],
            'cold_empath':[normalized],
            'competing_empath':[normalized],
            'vacation_empath':[normalized],
            'work_empath':[normalized],
            'real_estate_empath':[normalized],
            'sleep_empath':[normalized],
            'cheerfulness_empath':[normalized],
            'fire_empath':[normalized],
            'help_empath':[normalized],
            'superhero_empath':[normalized],
            'fun_empath':[normalized],
            'blue_collar_job_empath':[normalized],
            'music_empath':[normalized],
            'fashion_empath':[normalized],
            'sentiment':[normalized_negative],
            'tweet number':[positive(),integer],
            'retweet number':[positive(),integer],
            'c_favorites_count':[positive()],
            'c_sentiment':[normalized_negative],
            'c_status length':[positive()],
            'c_baddies':[positive()],
            'c_mentions':[positive()],
            'c_noise_empath':[normalized],
            'c_magic_empath':[normalized],
            'c_politics_empath':[normalized],
            'c_lust_empath':[normalized],
            'c_communication_empath':[normalized],
            'c_sleep_empath':[normalized],
            'c_leisure_empath':[normalized],
            'c_affection_empath':[normalized],
            'c_children_empath':[normalized],
            'c_rage_empath':[normalized],
            'c_hipster_empath':[normalized],
            'c_negotiate_empath':[normalized],
            'c_work_empath':[normalized],
            'c_social_media_empath':[normalized],
            'c_hygiene_empath':[normalized],
            'c_aggression_empath':[normalized],
            'c_exercise_empath':[normalized],
            'c_dispute_empath':[normalized],
            'c_music_empath':[normalized],
            'c_ridicule_empath':[normalized],
            'c_warmth_empath':[normalized],
            'c_prison_empath':[normalized],
            'c_terrorism_empath':[normalized],
            'c_real_estate_empath':[normalized],
            'c_furniture_empath':[normalized],
            'c_liquid_empath':[normalized],
            'c_dominant_heirarchical_empath':[normalized],
            'c_school_empath':[normalized],
            'c_joy_empath':[normalized],
            'c_cleaning_empath':[normalized],
            'c_government_empath':[normalized],
            'c_vehicle_empath':[normalized],
            'c_money_empath':[normalized],
            'c_science_empath':[normalized],
            'c_superhero_empath':[normalized],
            'c_fire_empath':[normalized],
            'c_home_empath':[normalized],
            'c_rural_empath':[normalized],
            'c_fight_empath':[normalized],
            'c_alcohol_empath':[normalized],
            'c_death_empath':[normalized],
            'c_shopping_empath':[normalized],
        }

    elif dataset_name == 'HCDR':
        constraints = {
            'NAME_CONTRACT_TYPE': [binary],
            'FLAG_OWN_CAR': [binary],
            'FLAG_OWN_REALTY': [binary],
            'AMT_INCOME_TOTAL':[positive()],
            'AMT_CREDIT':[positive()],
            'NAME_TYPE_SUITE':[categorical(0, 7)],
            'NAME_INCOME_TYPE':[categorical(0, 8)],
            'NAME_HOUSING_TYPE':[categorical(0, 6)],
            'DAYS_EMPLOYED':[negative(), integer],
            'DAYS_REGISTRATION':[negative(), integer],
            'DAYS_ID_PUBLISH':[negative(), integer],
            'OWN_CAR_AGE':[positive(), integer],
            'FLAG_WORK_PHONE': [binary],
            'FLAG_PHONE': [binary],
            'FLAG_EMAIL': [binary],
            'OCCUPATION_TYPE':[categorical(0, 18)],
            'REGION_RATING_CLIENT':[categorical(0, 3)], #?
            'WEEKDAY_APPR_PROCESS_START':[categorical(0, 7)],
            'LIVE_REGION_NOT_WORK_REGION':[binary],
            'REG_CITY_NOT_LIVE_CITY':[binary],
            'LIVE_CITY_NOT_WORK_CITY':[binary],
            'ORGANIZATION_TYPE':[categorical(0, 57)],
            'WALLSMATERIAL_MODE':[categorical(0, 7)],
            'DAYS_LAST_PHONE_CHANGE':[negative(), integer],
            'DEF_30_CNT_SOCIAL_CIRCLE':[DEF_30_CNT_SOCIAL_CIRCLE,positive(), integer],
            'DEF_60_CNT_SOCIAL_CIRCLE':[DEF_60_CNT_SOCIAL_CIRCLE,positive(), integer],
            'EXT_SOURCE_1':[EXT_SOURCE_1, normalized],
            'EXT_SOURCE_2':[EXT_SOURCE_2, normalized],
            'EXT_SOURCE_3':[EXT_SOURCE_3, normalized],
        }
    
    elif dataset_name == "ICU":
        constraints = {
            'hospital_id':[positive(1),integer],	
            'hospital_admit_source':[integer,categorical(0,14)],
            #'weight':[positive(), increas(2)],
            'bun_apache':[positive()],
            'gcs_verbal_apache':[integer,categorical(1,5)],
            'glucose_apache':[positive(),integer],
            'heart_rate_apache':[positive(1),integer],
            'resprate_apache':[positive()],
            'sodium_apache':[positive()],
            'urineoutput_apache':[positive()],
            'd1_diasbp_invasive_max':[positive(1),integer],
            'd1_diasbp_invasive_min':[positive(1),integer],
            'd1_diasbp_max':[positive(1),integer],
            'd1_diasbp_min':[positive(1),integer],
            'd1_heartrate_max':[positive(1),integer],
            'd1_heartrate_min':[positive(1),integer],
            'd1_mbp_max':[positive(),integer],
            'd1_mbp_min':[positive(),integer],
            'd1_resprate_max':[positive(),integer],
            'd1_resprate_min':[positive(),integer],
            'd1_sysbp_invasive_max':[positive(1),integer],
            'd1_sysbp_invasive_min':[positive(1),integer],
            'd1_sysbp_max':[positive(),integer],
            'd1_sysbp_min':[positive(),integer],
            'h1_diasbp_max':[positive(),integer],
            'h1_diasbp_min':[positive(),integer],
            'h1_heartrate_max':[positive(),integer],
            'h1_heartrate_min':[positive(),integer],
            'h1_mbp_max':[positive(),integer],
            'h1_mbp_min':[positive(),integer],
            'h1_resprate_max':[positive(),integer],
            'h1_resprate_min':[positive(),integer],
            'h1_sysbp_max':[positive(),integer],
            'h1_sysbp_min':[positive(),integer],
            'd1_glucose_min':[positive(1),integer],
            'd1_sodium_max':[positive(),integer],
            'd1_sodium_min':[positive(),integer],
            'h1_glucose_max':[positive(),integer],
            'h1_sodium_max':[positive(),integer],
            #'bmi':[bmi],
            'apache_3j_bodysystem':[apache_3j_bodysystem,integer,categorical(0,10)],
            'apache_3j_diagnosis':[apache_3j_diagnosis,positive()],
            'd1_mbp_invasive_max':[d1_mbp_invasive_max,positive(1),integer],
            'd1_mbp_invasive_min':[d1_mbp_invasive_min,positive(1),integer],
        }
        
    else: #RADCOM

        constraints = {
            'agg_count':[positive(1,300),integer], 
            'delta_delta_delta_from_previous_request':[positive(0,1000),integer], #1
            'delta_delta_from_previous_request':[positive(0,1000),integer],  # 2
            'delta_from_previous_request':[positive(0,1000),integer],   # 3
            'delta_from_start': [positive(0,1000),integer],   # 4
            'effective_peak_duration': [positive(0,1000),integer],   # 100000, 0.01  # 5
            # 'index':range(), # 6
            # 'minimal_bit_rate':range(), # 7
            'non_request_data': [positive(0,1000),integer],  # 8
            # 'peak_duration':range(), # 9
            # 'peak_duration_sum':range(), # 10
            'previous_previous_previous_previous_total_sum_of_data_to_sec': [ positive(0,100000)],  # 11
            'previous_previous_previous_total_sum_of_data_to_sec':[positive(0,100000)],  # 12
            'previous_previous_total_sum_of_data_to_sec': [positive(0,100000)],  # 13
            'previous_total_sum_of_data_to_sec': [positive(0,100000)],  # 14
            'sum_of_data': [positive(0,100000),integer],  # 100000000, 1 # 15
            'total_sum_of_data':[positive(0,100000),integer], # 100000000, 1 # 16
            'total_sum_of_data_to_sec': [positive(0,100000)],  # 1000000, 1 # 17
            'serv_label': [categorical(0,3),integer],  # 0,1,2 # 18
            #'start_of_peak_date': date_change(),  # 19
            #'start_of_peak_time': date_change(),  # 20
            #'end_of_peak_date': time_change(),  # 21
            #'end_of_peak_time': time_change(),  # 22 
        }

    return constraints, perturbability


def get_hopskipjump(classifier, params):
    hsj = HopSkipJump(classifier=classifier, 
                          **params,
                        )
    return hsj

def get_boundary(classifier, params): #epsilon,delta, max_iter, num_trial):
    bnd = BoundaryAttack( estimator=classifier, 
                         **params,
                         )
    return bnd

def get_surrogate_Attack(classifier, params): #columns_names):
    sa = SurrogateAttack(classifier=classifier,
                            **params,
                        )
    return sa


def get_attack_set(datasets, target_models, surr_model, scaler ,data_path):
    
    files = os.listdir(data_path)
    if "x_attack.csv" in files :
        attack_x = pd.read_csv(data_path + "/x_attack_clean.csv")
        attack_y = pd.read_csv(data_path + "/y_attack_clean.csv")
        return attack_x, attack_y


    attack_x = datasets.get("x_test")
    attack_y = datasets.get("y_test")

    # 1 for samples predicted correct by all models
    equal_preds = np.ones_like(attack_y.to_numpy().T)

    #extract only samples that predicted correct by surrogate
    if surr_model != None:
        pred_original_surr = torch.sigmoid(surr_model(torch.from_numpy(scaler.transform(attack_x)).float())).round().detach().numpy().reshape(1,-1)
        equal_preds[pred_original_surr != attack_y.T] = 0

    #extract only samples that predicted correct by all target models
    for j, target in enumerate(target_models):
        pred_original_target = target.predict(attack_x.to_numpy())
        equal_preds[pred_original_target != attack_y.T] = 0
    
    attack_x_clean = attack_x.iloc[np.where(equal_preds[0] == 1)]
    attack_y_clean = attack_y.iloc[np.where(equal_preds[0] == 1)]
    attack_x_clean.to_csv(data_path + '/x_attack_clean',index=False)
    attack_y_clean.to_csv(data_path + '/y_attack_clean',index=False)

    return attack_x_clean, attack_y_clean

def get_adv_init(datasets, target_models, surr_model, scaler ,data_path):
    
    files = os.listdir(data_path)
    if "x_attack.csv" in files :
        x_adv_init = pd.read_csv(data_path + "/x_adv_init.csv")
        y_adv_init = pd.read_csv(data_path + "/y_adv_init.csv")
        return x_adv_init, y_adv_init


    x_adv_init = datasets.get("x_train_surrogate")
    y_adv_init = datasets.get("y_train_surrogate")

    # 1 for samples predicted correct by all models
    equal_preds = np.ones_like(y_adv_init.to_numpy().T)

    #extract only samples that predicted correct by surrogate
    pred_original_surr = torch.sigmoid(surr_model(torch.from_numpy(scaler.transform(x_adv_init)).float())).round().detach().numpy().reshape(1,-1)
    equal_preds[pred_original_surr != y_adv_init.T] = 0

    #extract only samples that predicted correct by all target models
    for j, target in enumerate(target_models):
        pred_original_target = target.predict(x_adv_init.to_numpy())
        equal_preds[pred_original_target != y_adv_init.T] = 0
    
    x_adv_init_clean = x_adv_init.iloc[np.where(equal_preds[0] == 1)]
    y_adv_init_clean = y_adv_init.iloc[np.where(equal_preds[0] == 1)]
    x_adv_init_clean.to_csv(data_path + '/x_adv_init.csv',index=False)
    y_adv_init_clean.to_csv(data_path + '/y_adv_init.csv',index=False)

    return x_adv_init_clean, y_adv_init_clean

def get_balanced_attack_set(dataset_name, attack_x, attack_y, attack_size=5, seed=42):
     # create balanced attack set
    attack_0 = attack_x[attack_y.pred == 0]
    attack_1 = attack_x[attack_y.pred == 1]
    attack_0= shuffle(attack_0, random_state=12)
    if not 'HATE' in dataset_name:
        attack_0 = attack_0.iloc[:attack_1.shape[0]]    
    
    '''
    # Get importance values
    importance = generate_params['importance_values']
    if (importance == 'LGB'):
        importance_values = LGB.feature_importances_
    elif (importance == 'RF'):
        importance_values = RF.feature_importances_
    elif (importance == 'GB'):
        importance_values = GB.feature_importances_
    elif (importance == 'XGB'):
        importance_values = XGB.feature_importances_
    
    generate_params['importance_values'] = importance_values
    '''

    if ('HATE' in dataset_name):
        if attack_size < attack_1.shape[0]:
            attack_1 = attack_1.iloc[:attack_size]
        attack_0, attack_0_ = train_test_split(attack_0, train_size=5*attack_1.shape[0], random_state=seed, shuffle=True)
    else:
        attack_0, attack_0_, attack_1, attack_1_ = train_test_split(attack_0, attack_1, train_size=attack_size, random_state=seed, shuffle=True)
    attack_x = pd.concat([attack_0, attack_1], ignore_index=True)

    attack_y = pd.DataFrame(0, index=range(attack_0.shape[0]), columns=['pred'])
    attack_y = pd.concat([attack_y, pd.DataFrame(1, index=range(attack_1.shape[0]), columns=['pred'])])    
    #attack_x = pd.concat([attack_x.iloc[:5],attack_x.iloc[-5:]], ignore_index=True)
    #attack_y = pd.concat([attack_y.iloc[:5],attack_y.iloc[-5:]], ignore_index=True)

    #return attack_x.iloc[500:], attack_y.iloc[500:]
    return attack_x, attack_y

def load_attacks(results_path, target_models_names):
    boundary_attack_name = '_delta-{}_epsilon-{}_num_trial-{}_max_iter-{}'.format(1.0,1.0,20,3000)
    hopskip_attack_name = '_max_iter-{}_max_eval-{}_init_size-{}'.format(50,10000,100)
    #surrogate_attack_name = '_max_iter-{}_max_eval-{}_init_size-{}'.format(1000,10000,100)
    
    attacks = {}
    
    for target_model in (target_models_names):
        #attacks ['surrogate_'+target_model] = pd.read_csv(results_path + '/surrogate/surrogate_samples_'+target_model+'-imp_'+surrogate_attack_name+'.csv').drop(['true_label'], axis=1)  
        attacks ['boundary_base_'+target_model] = pd.read_csv(results_path + '/boundary_base/boundary_base_samples_'+target_model+boundary_attack_name+'.csv').drop(['true_label'], axis=1)
        #attacks ['boundary_tabular_'+target_model] = pd.read_csv(results_path + '/boundary_tabular/tabular_boundary_samples_'+target_model+boundary_attack_name+'.csv').drop(['true_label'], axis=1)
        attacks ['hopskip_base_'+target_model] = pd.read_csv(results_path + '/hopskip_base/hopskip_base_samples_'+target_model+hopskip_attack_name+'.csv').drop(['true_label'], axis=1)  
        attacks ['hopskip_tabular_'+target_model] = pd.read_csv(results_path + '/hopskip_tabular/tabular_hopskip_samples_'+target_model+hopskip_attack_name+'.csv').drop(['true_label'], axis=1)
           
    return attacks
