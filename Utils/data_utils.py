import os
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.utils import resample, shuffle
import torch

# Import IterativeImputer from fancyimpute
#from fancyimpute import IterativeImputer


def process(dataset_name, raw_data_path, TorchMinMaxScaler):
    """
    # Pre-Proccess data
    preprocess_HCDR(raw_data_path)
    print(x)
    preprocess_ICU(raw_data_path)
    preprocess_HATE(raw_data_path)
    preprocess_top100(raw_data_path, dataset_name)
    
    write_edditable_file(raw_data_path)
    
    """
    """
    # Rearange data after proccessing and constrance
    #rearrange columns: edittable first
    rearrange_columns(raw_data_path, dataset_name, perturbability_path)
    rearrange_columns_edittable(perturbability_path)
    
    """
    # Set a scaler for thr data
    #data = pd.read_csv(raw_data_path+"/after_preprocessing/"+dataset_name+"_after_preprocessing_no_glove_top_115_rearrange.csv")
    
    data = pd.read_csv(raw_data_path+"/after_preprocessing/"+dataset_name+"_after_preprocessing_rearrange.csv")
    data = data.drop('pred', axis =1)
    data1 = torch.tensor(np.array(data), requires_grad=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data.to_numpy())
    df = scaler.transform(data)
    pickle.dump(scaler, open(raw_data_path + "/scaler_pt.pkl", 'wb'))
    
    scaler_t = TorchMinMaxScaler(feature_range=(0, 1))
    scaler_t.fit(data1)
    df_t = scaler_t.transform(data1)
    print(np.equal(df, df_t.detach().numpy()))
    pickle.dump(scaler_t, open(raw_data_path + "/scaler_pt.pkl", 'wb'))

def split_to_datasets(raw_data_path, seed=42, val_size=0.25, surrgate_train_size=0.5, save_path=None, exclude=None):
    files = os.listdir(save_path)
    if "x_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size) in files:
        x_train_target = pd.read_csv(save_path + "/x_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        x_train_surrogate = pd.read_csv(save_path + "/x_train_surrogate_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_train_target = pd.read_csv(save_path + "/y_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_train_surrogate = pd.read_csv(save_path + "/y_train_surrogate_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        x_test = pd.read_csv(save_path + "/x_test_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_test = pd.read_csv(save_path + "/y_test_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        datasets = {
            "x_train_target": x_train_target,
            "x_train_surrogate": x_train_surrogate,
            "y_train_target": y_train_target,
            "y_train_surrogate": y_train_surrogate,
            "x_test": x_test,
            "y_test": y_test
        }
        return datasets

    if "x_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size) in files and exclude is None:
        x_train_target = pd.read_csv(save_path + "/x_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        x_train_surrogate = pd.read_csv(save_path +
            "/x_train_surrogate_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
                seed, val_size, surrgate_train_size))
        y_train_target = pd.read_csv(save_path + "/y_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        y_train_surrogate = pd.read_csv(save_path +
            "/y_train_surrogate_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
                seed, val_size, surrgate_train_size))
        x_test = pd.read_csv(save_path + "/x_test_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        y_test = pd.read_csv(save_path + "/y_test_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        datasets = {
            "x_train_target": x_train_target,
            "x_train_surrogate": x_train_surrogate,
            "y_train_target": y_train_target,
            "y_train_surrogate": y_train_surrogate,
            "x_test": x_test,
            "y_test": y_test
        }
        return datasets
    
    data_raw = pd.read_csv(raw_data_path+'/after_preprocessing/HCDR_after_preprocessing_rearrange.csv').reset_index(drop=True) #_rearrange
    if "Unnamed: 0" in data_raw.columns:
        data_raw = data_raw.drop(["Unnamed: 0"], axis=1)
    
    #data_raw = data_raw.drop(['hate_neigh'], axis=1)
    
    
    #if raw_data_path.split("/")[1] == "HATE":
    #    data_raw["hate_neigh"] = data_raw["hate_neigh"].apply(lambda x: int(x))

    if exclude is not None:
        data_raw = data_raw.drop(exclude, axis=1)

    # balance the data AMIT
    if raw_data_path.split("/")[1] in {"H","I"}:
        g = data_raw.groupby('pred')
        data_raw = g.apply(lambda x: x.sample(g.size().max(), replace=True).reset_index(drop=True))

        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_test, y_train, y_test = train_test_split(data_raw_x,
                                                            data_raw_y,
                                                            test_size=val_size,
                                                            random_state=seed)
        x_train_target, x_train_surrogate, y_train_target, y_train_surrogate = train_test_split(x_train,
                                                                                                y_train,
                                                                                                random_state=seed)
    # balance the data YAEL 
    if raw_data_path.split("/")[1] in {"HATE","ICU","HCDR","RADCOM"}:
        # Separate majority and minority classes
        df_majority = data_raw[data_raw.pred==0]
        df_minority = data_raw[data_raw.pred==1]

        Train_maj, Test_maj = train_test_split(df_majority, test_size = val_size, random_state = seed)
        Train_min, Test_min = train_test_split(df_minority, test_size = val_size, random_state = seed)


        # Resampling the minority levels to match the majority level
        # Upsample minority class
        df_minority_upsampled = resample(Train_min, 
                                        replace=True,     # sample with replacement
                                        n_samples=Train_maj.shape[0],    # to match majority class
                                        random_state= seed) # reproducible results
        
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([Train_maj, df_minority_upsampled]) #classes are equals now
        test =  pd.concat([Test_maj, Test_min])

        train_target, train_surrogate = train_test_split(df_upsampled, 
                                                        test_size = surrgate_train_size,
                                                        shuffle=True, 
                                                        random_state = seed)
        x_train_target = train_target.copy().drop('pred', axis = 1)
        y_train_target = train_target['pred']
        x_train_surrogate = train_surrogate.copy().drop('pred', axis = 1)
        y_train_surrogate = train_surrogate['pred']
        x_test = test.copy().drop('pred', axis = 1)
        y_test = test['pred']
        '''
        x_train_target.set_index('encounter_id', inplace = True)
        y_train_target.set_index('encounter_id', inplace = True)
        x_train_surrogate.set_index('encounter_id', inplace = True)
        y_train_surrogate.set_index('encounter_id', inplace = True)
        x_test.set_index('encounter_id', inplace = True)
        y_test.set_index('encounter_id', inplace = True)
        '''                                                                                                             
    else:
        data_raw_x = data_raw.copy().drop('pred', axis=1)
        data_raw_y = data_raw.copy()['pred']
        x_train, x_test, y_train, y_test = train_test_split(data_raw_x,
                                                            data_raw_y,
                                                            test_size=val_size,
                                                            random_state=seed)
        x_train_target, x_train_surrogate, y_train_target, y_train_surrogate = train_test_split(x_train,
                                                                                                y_train,
                                                                                                test_size=surrgate_train_size,
                                                                                                random_state=seed)
                                                                                                  
    datasets = {
        "x_train_target": x_train_target,
        "x_train_surrogate": x_train_surrogate,
        "y_train_target": y_train_target,
        "y_train_surrogate": y_train_surrogate,
        "x_test": x_test,
        "y_test": y_test
    }

    if save_path is not None:
        for key in datasets.keys():
            if exclude is not None:
                file_name = str(key) + "_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(exclude,seed, val_size, surrgate_train_size)
            else:
                file_name = str(key) + "_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(seed, val_size, surrgate_train_size)
            cur_saving_path = save_path + "/" + file_name
            datasets.get(key).to_csv(cur_saving_path, index=False)

    
    return datasets

def over_sampling(x_data, y_data, seed=42):
    data_raw = x_data.copy()
    data_raw['pred'] = y_data.copy() 
    data_raw = shuffle(data_raw)
    '''
    # AMIT                                                                            
    g = data_raw.groupby('pred')
    x_data = g.apply(lambda x: x.sample(g.size().max(), replace=True).reset_index(drop=True))
    x_data = data_raw.drop("pred", axis=1)
    y_data = pd.DataFrame(data_raw["pred"])
    '''
    # Separate majority and minority classes
    df_majority = data_raw[data_raw.pred==0]
    df_minority = data_raw[data_raw.pred==1]

    # Resampling the minority levels to match the majority level
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=df_majority.shape[0],    # to match majority class
                                    random_state= seed) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled]) #classes are equals now
    y_row= pd.DataFrame(df_upsampled['pred'])
    x_row = df_upsampled.drop('pred', axis=1)
    return x_row, y_row


def rearrange_columns(raw_data_path, dataset_name, perturbability_path):
    edittability = pd.read_csv(perturbability_path)
    perturbability = edittability["Perturbability"].to_numpy()
    features = edittability["feature"].to_numpy()
    data_raw = pd.read_csv(raw_data_path+'/after_preprocessing/HCDR_after_preprocessing.csv')
    pred = data_raw['pred']
    data_raw = data_raw.drop(['pred'], axis=1)
    new_f_1 = features[perturbability == 1]
    new_f_0 = features[perturbability == 0]
    ind = np.append(new_f_1, new_f_0)
    new_df = pd.DataFrame(data_raw, columns=ind)
    new_df['pred'] = pred
    new_df.to_csv(raw_data_path+ "/after_preprocessing/HCDR_after_preprocessing_rearrange.csv",index=False)

def rearrange_columns_edittable(perturbability_path):
    edittability = pd.read_csv(perturbability_path)
    perturbability = edittability["Perturbability"].to_numpy()
    features = edittability["feature"].to_numpy()
    data_raw = perturbability.copy()
    new_f_1 = features[perturbability == 1]
    new_f_0 = features[perturbability == 0]
    ind = np.append(new_f_1, new_f_0)
    new_df = pd.DataFrame(columns=["feature","Perturbability"])
    new_df["feature"] = ind
    for i, _feature in enumerate (new_df["feature"]):
        index = np.where(features == _feature)[0]
        new_df["Perturbability"][i] = perturbability[index][0]
    new_df.to_csv("Datasets/HCDR" + "/edittible_features_rearrange.csv",index=False)

def write_edditable_file(init_data_path):
    #df = pd.read_csv(init_data_path+ "/after_preprocessing/HCDR_after_preprocessing_yael_no_missing.csv")
    preds = df['pred']
    final = df.drop(['pred'], axis=1)
    features = final.columns.to_frame()
    features.to_csv(init_data_path +'/edittible_features.csv', index=False)

def drop_corolated_all(df):
    
    #df = shuffle(df)
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.9
    to_drop_90 = [column for column in upper.columns if any(upper[column] > 0.9)]

    return df.drop(to_drop_90, axis=1)

def drop_corolated_target(df):
    
    pred = df["pred"]
    df = df.drop('pred', axis=1)
    df.insert(loc=0, column='pred', value=pred)

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.9
    to_drop_90 = [column for column in upper.columns if any(upper[column] > 0.9)]

    return df.drop(to_drop_90, axis=1)

def remove_nulls_and_impute(df, target_name=None):

    ## remove features with missing values
    if target_name!= None:
        labels = df[target_name]
        df = df.drop(columns=[target_name])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace([np.inf, -np.inf,'NA'], np.nan, inplace=True)

    NA_col_train = pd.DataFrame(df.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(df))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    df = df.drop(cols_to_drop, axis=1)

    ## impute
    imputer_numeric = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    numeric_cols= list(set(df.columns) - set(cat_cols))
    
    imputer_numeric.fit(df[numeric_cols])
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    if len(cat_cols) > 0:
        imputer_cat.fit(df[cat_cols])
        df[cat_cols] = imputer_cat.transform(df[cat_cols])
    
    if target_name!= None:
        df[target_name] = labels
    return df

# ___________________________________________________ HCDR ________________________________________________________

def get_age_group(days_birth):
    age_years = -days_birth / 365
    if age_years < 27:
        return 1
    elif age_years < 40:
        return 2
    elif age_years < 50:
        return 3
    elif age_years < 65:
        return 4
    elif age_years < 99:
        return 5
    else:
        return 0


def do_mean(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_median(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_std(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def label_encoder(df, categorical_columns=None):
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col], use_na_sentinel=True)
    return df, categorical_columns


def drop_application_columns(df):
    drop_list = [
        'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def group_f(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = group_f(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)


def get_bureau_balance(path, num_rows= None):
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'))
    bb.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'X': np.nan}, inplace = True)
    bb = remove_nulls_and_impute(bb)
    #bb, categorical_cols = label_encoder(bb)
    # Calculate rate for each category with decay
    categorical_cols = [col for col in bb.columns if bb[col].dtype == 'object']
    bb_processed = bb.groupby('SK_ID_BUREAU')[categorical_cols].agg(lambda x: x.value_counts().index[0]).reset_index()
    # Min, Max, Count and mean duration of payments (months)
    agg = {'MONTHS_BALANCE': ['size']}
    bb_processed = group_and_merge(bb, bb_processed, '', agg, 'SK_ID_BUREAU')
    return bb_processed, categorical_cols

def remove_missing_rows(df, target_name):
    labels = df['TARGET']
    df = df.drop(columns=['TARGET'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace([np.inf, -np.inf,'NA'], np.nan, inplace=True)

    NA_col_train = pd.DataFrame(df.isna().sum(axis=1), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/df.shape[1])*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 49]
    cols_to_drop = NA_col_train.index.tolist()
    df = df.drop(cols_to_drop, axis=0)

    df[target_name] = labels
    return df

def preprocess_HCDR(init_data_path):
    # Taken from https://www.kaggle.com/code/lolokiller/model-with-bayesian-optimization
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.
    """
    df = pd.read_csv(init_data_path + "/original/application_train.csv")

    # Remove outliers
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    #YAEL
    df.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True)
    df = remove_missing_rows(df, 'TARGET') # >= 49%
    df = remove_nulls_and_impute(df, 'TARGET')

    # Feature engineering
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    #df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_group(x))
    #df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    #df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    
    df = drop_application_columns(df)
    
    # Encoding categorical
    #df, le_encoded_cols = label_encoder(df, None)
    #cat_cols.extend(le_encoded_cols) # append to cat_cols
     
    # Add Bureau data
    bureau = pd.read_csv(init_data_path + "/original/bureau.csv")
    
    #YAEL
    bureau.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True)
    bureau = remove_nulls_and_impute(bureau)

    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    #bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    #bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
   
    #YAEL
    bureau = bureau.drop('DAYS_ENDDATE_FACT', axis=1)
    
    bureau_balance,  categorical_cols = get_bureau_balance(init_data_path+'/original')
    bureau = bureau.merge(bureau_balance, how='left', on='SK_ID_BUREAU')
    
    features = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM',
                'AMT_CREDIT_SUM_DEBT', 'DEBT_CREDIT_DIFF']#, 'STATUS_0', 'STATUS_12345']
    agg_length = bureau.groupby('MONTHS_BALANCE_SIZE')[features].mean().reset_index()
    agg_length.rename({feat: 'LL_' + feat for feat in features}, axis=1, inplace=True)
    bureau = bureau.merge(agg_length, how='left', on='MONTHS_BALANCE_SIZE')
    
    BUREAU_AGG = {
        'SK_ID_BUREAU': ['nunique'],
        'AMT_CREDIT_SUM': ['sum'],# ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],#['max', 'mean', 'sum'],
      
    }
    BUREAU_ACTIVE_AGG = {
        'AMT_CREDIT_SUM': ['sum'],#['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],#['mean', 'sum'],
    }
    
    agg_bureau = group_f(bureau, 'BUREAU_', BUREAU_AGG)
    
    #YAEL
    active = bureau.copy()
    active.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True)
    active = remove_nulls_and_impute(active)

    active = active[active['STATUS'] != 'C']
    agg_bureau = group_and_merge(active, agg_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
    
    sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])
    gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()
    gr.rename({'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_LAST_LOAN_MAX_OVERDUE'}, inplace=True)
    agg_bureau = agg_bureau.merge(gr, on='SK_ID_CURR', how='left')
    '''
    agg_bureau['BUREAU_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
    agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
    agg_bureau = agg_bureau.drop(['BUREAU_AMT_CREDIT_SUM_DEBT_SUM','BUREAU_AMT_CREDIT_SUM_SUM','BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM','BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM' ], axis=1)
    '''
    agg_bureau.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True)
    agg_bureau = remove_nulls_and_impute(agg_bureau)
    #agg_bureau, categorical_cols = label_encoder(agg_bureau)
    #cat_cols.extend(categorical_cols) # append to cat_cols
    df = pd.merge(df, agg_bureau, on='SK_ID_CURR', how='left')
    
    # Adding Installments/ previous application data
    prev = pd.read_csv(os.path.join(init_data_path, 'original/previous_application.csv'))
    pay = pd.read_csv(os.path.join(init_data_path, 'original/installments_payments.csv'))
    
    #YAEL
    prev.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True)
    prev_drop = ['SK_ID_PREV','HOUR_APPR_PROCESS_START', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'CODE_REJECT_REASON', 'NAME_GOODS_CATEGORY', 'PRODUCT_COMBINATION']
    prev = prev.drop(prev_drop, axis=1)
    prev = remove_nulls_and_impute(prev)

    #prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    #prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    #prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    #prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    #total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    #prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].copy()
    
    # YAEL
    #approved_gry1 = approved.loc[approved.groupby(["SK_ID_CURR"])["AMT_CREDIT"].idxmax()] #take longer
    approved_gr = approved.sort_values('AMT_CREDIT', ascending=False).drop_duplicates(['SK_ID_CURR'])
    sk_id = approved_gr['SK_ID_CURR']
    approved_gr = approved_gr.drop('SK_ID_CURR',axis=1)
    approved_gr.columns = pd.Index(['{}_{}'.format('PREV_APPROVE', e)
                               for e in approved_gr.columns.tolist()])
    approved_gr['SK_ID_CURR'] = sk_id

    refused_gr = refused.sort_values('AMT_CREDIT', ascending=True).drop_duplicates(['SK_ID_CURR'])
    sk_id = refused_gr['SK_ID_CURR']
    refused_gr = refused_gr.drop('SK_ID_CURR',axis=1)
    refused_gr.columns = pd.Index(['{}_{}'.format('PREV_REFUSE', e)
                               for e in refused_gr.columns.tolist()])
    refused_gr['SK_ID_CURR'] = sk_id

     
    #approved_gr, categorical_cols = label_encoder(approved_gr)
    #cat_cols.extend(categorical_cols) # append to cat_cols
    #refused_gr, categorical_cols = label_encoder(refused_gr)
    #cat_cols.extend(categorical_cols) # append to cat_cols

    df = pd.merge(df, approved_gr, on='SK_ID_CURR', how='left')
    df = pd.merge(df, refused_gr, on='SK_ID_CURR', how='left')

    df.to_csv("/".join(init_data_path.split("/")[:-1]) + "/HCDR/after_preprocessing/HCDR_after_preprocessing_with_orig_category.csv",
                 index=False)
    """
    ## YAEL post-process ##
    df = pd.read_csv(init_data_path + "/after_preprocessing/HCDR_after_preprocessing_with_orig_category.csv")
    # Fill NA and scale
    labels = df['TARGET']
    df = df.drop(columns=["TARGET"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan, 'XAP': np.nan}, inplace = True) #XAP

    NA_col_train = pd.DataFrame(df.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(df))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    df = df.drop(cols_to_drop, axis=1)
    
    df = df.drop('SK_ID_CURR', axis=1)

    #df = df.drop('CODE_GENDER', axis=1)

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
   
    # impute nan
    imputer_numeric = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    numeric_cols= list(set(df.columns) - set(cat_cols))
    imputer_numeric.fit(df[numeric_cols])
    imputer_cat.fit(df[cat_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    #pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))
    pickle.dump(imputer_cat, open(init_data_path + "/imputer_cat_orig.pkl", 'wb'))
    pickle.dump(imputer_numeric, open(init_data_path + "/imputer_numeric.pkl", 'wb'))
    
    # binary features
    bin_cols = [col for col in df.columns if 'Y' in df[col].values]
    df_copy = df.copy()
    df_copy[bin_cols].astype(str)
    for col in bin_cols:
        df_copy[col] = pd.Series(np.where(df_copy[col].values == 'Y', 1, 0), df_copy.index)
   
    #df[bin_cols].astype(str).astype(int)
    df[bin_cols] = df_copy[bin_cols]
    df[bin_cols].astype(object)

    #categoric_cols = [col for col in cat_cols if col in df.columns]
    categoric_cols = [col for col in cat_cols if (col not in bin_cols)]
    df, _ = label_encoder(df, categoric_cols)

    imputer_cat.fit(df[cat_cols])
    pickle.dump(imputer_cat, open(init_data_path + "/imputer_cat_encode.pkl", 'wb'))
   
    df[bin_cols].astype(str).astype(int)

    colonnes = df.columns
    feature = list(df.columns)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    final["pred"] = labels
    final = drop_corolated_target(final)
    
    final = final.drop(['BUREAU_DEBT_OVER_CREDIT','BUREAU_ACTIVE_DEBT_OVER_CREDIT'], axis=1) #BUREAU_DEBT_OVER_CREDIT,0

    final = final.drop(['pred'], axis=1)
    features = final.columns.to_frame()
    features.to_csv(init_data_path +'/edittible_features.csv', index=False)

    final["pred"] = labels
    final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/HCDR/after_preprocessing/HCDR_after_preprocessing.csv",
                 index=False)
    

# ___________________________________________________ ICU ________________________________________________________

def preprocess_HCDR_(init_data_path):
    # Taken from https://www.kaggle.com/code/binaicrai/fork-of-fork-of-wids-lgbm-gs
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    train = pd.read_csv(init_data_path , "/original/application_train.csv")
    train = pd.read_csv(init_data_path , "/original/bureau.csv")
    # Adding Installments/ previous application data
    prev = pd.read_csv(os.path.join(init_data_path, '/original/previous_application.csv'))
    pay = pd.read_csv(os.path.join(init_data_path, '/original/installments_payments.csv'))

    train_len = len(train)
    combined_dataset = pd.concat(objs = [train, test], axis = 0)

    # cleanlab works with **any classifier**. Yup, you can use sklearn/PyTorch/TensorFlow/XGBoost/etc.
    cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

    # cleanlab finds data and label issues in **any dataset**... in ONE line of code!
    label_issues = cl.find_label_issues(data, labels)

    # cleanlab trains a robust version of your model that works more reliably with noisy data.
    cl.fit(data, labels)

    # cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
    cl.predict(test_data)

    # A true data-centric AI package, cleanlab quantifies class-level issues and overall data quality, for any dataset.
    cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)

def preprocess_top100(init_data_path, data):
    train = pd.read_csv(init_data_path + "/after_preprocessing/{}_after_preprocessing_impute_80.csv", data)
    labels = train["pred"]
    train = train.drop(['pred'], axis=1)

    # drop not important
    #drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    #final = final.drop(drop_list, axis=1)

    # take top 150
    #top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
    #    "feature"].values.tolist()
    #final = final[top_list]

    final["pred"] = labels
    #final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",
    #             index=False)
    final.to_csv(init_data_path+ "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",index=False)
    

def preprocess_ICU(init_data_path):
    # Taken from https://www.kaggle.com/code/binaicrai/fork-of-fork-of-wids-lgbm-gs
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    train = pd.read_csv(init_data_path + "/original/training_v2.csv")
    test = pd.read_csv(init_data_path + "/original/unlabeled.csv")
    train_len = len(train)
    combined_dataset = pd.concat(objs = [train, test], axis = 0)

    # Extracing categorical columns
    
    #df_cat = combined_dataset.select_dtypes(include=['object', 'category']) 

    '''
    'hospital_admit_source':
    Grouping: ['Other ICU', 'ICU']; ['ICU to SDU', 'Step-Down Unit (SDU)']; ['Other Hospital', 'Other']; ['Recovery Room','Observatoin']
    Renaming: Acute Care/Floor to Acute Care
    'icu_type':
    Grouping of the following can be explored: ['CCU-CTICU', 'CTICU', 'Cardiac ICU']
    'apache_2_bodysystem':
    Grouping of the following can be explored: ['Undefined Diagnoses', 'Undefined diagnoses']
    '''
    combined_dataset['hospital_admit_source'] = combined_dataset['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU',
                                                                                               'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})
    # combined_dataset['icu_type'] = combined_dataset['icu_type'].replace({'CCU-CTICU': 'Grpd_CICU', 'CTICU':'Grpd_CICU', 'Cardiac ICU':'Grpd_CICU'}) # Can be explored
    combined_dataset['apache_2_bodysystem'] = combined_dataset['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})   
    
    # Dropping few column/s with single value and all unique values
    # Dropping 'readmission_status', 'patient_id', along with 'gender'
    combined_dataset = combined_dataset.drop(['readmission_status', 'patient_id', 'gender'], axis=1)
    
    train = copy.copy(combined_dataset[:train_len])
    test = copy.copy(combined_dataset[train_len:])

    # Checking NAs for initial column clipping 
    # On train data
    pd.set_option('display.max_rows', 500)
    NA_col_train = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(train))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    # cols_to_drop.remove('hospital_death')
   
    # Dropping columns with >= 80% of NAs
    combined_dataset = combined_dataset.drop(cols_to_drop, axis=1)
    
    train = copy.copy(combined_dataset[:train_len])
    test = copy.copy(combined_dataset[train_len:])

    # MICE Imputation
    #7. MICE Imputation 
    #['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
    # Suggestion Courtesy: Bruno Taveres - https://www.kaggle.com/c/widsdatathon2020/discussion/130532
    # Adding 2 apache columns as well


    # Initialize IterativeImputer
    mice_imputer = IterativeImputer()

    # Impute using fit_tranform on diabetes
    train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']] = mice_imputer.fit_transform(train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']])
    
    # impute BMI = Weight(kg)/(Height(m)* Height(m))
    train['new_bmi'] = (train['weight']*10000)/(train['height']*train['height'])
    train['bmi'] = train['bmi'].fillna(train['new_bmi'])
    train = train.drop(['new_bmi'], axis = 1)

    # Extracting columns to change to Categorical
    col_train = train.columns
    l1 = []
    for i in col_train:
        if train[i].nunique() <= 16:
            l1.append(i)
                
    l1.remove('hospital_death')
    train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)

    cols = train.columns
    num_cols = train._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    # Fill NA
    df=train.copy()
    labels = df['hospital_death']
    df = df.drop(columns=["hospital_death"])
    colonnes = df.columns
    feature = list(df.columns)

    df.replace([np.inf, -np.inf,'NA'], np.nan, inplace=True)
    
    imputer_numeric = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    numeric_cols= list(set(num_cols) - set(['hospital_death']))
    imputer_numeric.fit(df[numeric_cols])
    imputer_cat.fit(df[cat_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    #pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))
    pickle.dump(imputer_cat, open(init_data_path + "/imputer_cat.pkl", 'wb'))
    pickle.dump(imputer_numeric, open(init_data_path + "/imputer_numeric.pkl", 'wb'))

    train = df
    for usecol in cat_cols:
        train[usecol] = train[usecol].astype('str')
        test[usecol] = test[usecol].astype('str')
        
        #Fit LabelEncoder
        le = LabelEncoder().fit(
                np.unique(train[usecol].unique().tolist()+ test[usecol].unique().tolist()))

        #At the end 0 will be used for dropped values
        train[usecol] = le.transform(train[usecol])+1
    
        train[usecol] = train[usecol].replace(np.nan, '').astype('int').astype('category')
       
    # SCALE
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    # drop not important
    #drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    #final = final.drop(drop_list, axis=1)

    # take top 150
    #top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
    #    "feature"].values.tolist()
    #final = final[top_list]

    final["pred"] = labels
    #final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",
    #             index=False)
    final.to_csv(init_data_path+ "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",index=False)
    
# ___________________________________________________ HATE ________________________________________________________
def preprocess_HATE(init_data_path):
    # Taken from 
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    try: # second preprocess if needed
        df = pd.read_csv(init_data_path+ "/after_preprocessing/HATE_after_preprocessing_no_glove.csv")
        preds = df['pred']
        top_features = np.array(pd.read_csv(init_data_path + "/importance/top_importance_no_glove.csv")).flatten()
        top_features = np.sort(top_features)
        #features = x_train_target.columns.to_frame()
       
        new_df = df.iloc[:,top_features]
        new_df['pred']= preds
        new_df.to_csv(init_data_path+ "/after_preprocessing/HATE_after_preprocessing_no_glove_top_117.csv",index=False)
        
        x = new_df.copy()
        x = drop_corolated_target(x)
        final = x.drop(['pred'], axis=1)
        final = drop_corolated_all(final)

        features = final.columns.to_frame()
        features.to_csv(init_data_path +'/edittible_features.csv', index=False)
        
        final['pred']= preds
        final.to_csv(init_data_path+ "/after_preprocessing/HATE_after_preprocessing_no_glove_top_115.csv",index=False)
        
        
        return

    except:
        pass

    train = pd.read_csv(init_data_path + "/original/users_neighborhood_anon.csv")
    train_len = len(train)
    
    #df_cat = combined_dataset.select_dtypes(include=['object', 'category']) 

    # Dropping few column/s with single value and all unique values
    # Dropping 'hashtags' (text values) and 'user_id'
    train = train.drop(['user_id','hashtags'], axis=1)
    
    # Removing glove features
    cols_glove = ["{0}_glove".format(v) for v in range(300)]
    cols_glove_c = ["c_" + v for v in cols_glove]

    train = train.drop(cols_glove, axis=1)
    train = train.drop(cols_glove_c, axis=1)
    
    '''
    # Checking NAs for initial column clipping 
    # On train data
    pd.set_option('display.max_rows', 500)
    NA_col_train = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(train))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    # cols_to_drop.remove('hospital_death')
   
    # Dropping columns with >= 80% of NAs
    combined_dataset = combined_dataset.drop(cols_to_drop, axis=1)
    '''

    # Extracting columns to change to Categorical
    col_train = train.columns
    l1 = []
    for i in col_train:
        if train[i].nunique() <= 16:
            l1.append(i)
                
    train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)

    cols = train.columns
    num_cols = train._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols)- set(['hate']))

    
    # Fill NA
    df=train.copy()
    labels = df['hate']
    df = df.drop(columns=["hate"])
    colonnes = df.columns
    feature = list(df.columns)

    #df.replace([np.inf, -np.inf,'NA'], np.nan, inplace=True)
    
    
    imputer_numeric = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    numeric_cols= list(set(num_cols))
    imputer_numeric.fit(df[numeric_cols])
    imputer_cat.fit(df[cat_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    #pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))
    pickle.dump(imputer_cat, open(init_data_path + "/imputer_cat.pkl", 'wb'))
    pickle.dump(imputer_numeric, open(init_data_path + "/imputer_numeric.pkl", 'wb'))

    train = df
    for usecol in cat_cols:
        #train[usecol] = train[usecol].astype('bool')
        train[usecol] = train[usecol].astype(int)
       
    # SCALE
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    # drop not important
    #drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    #final = final.drop(drop_list, axis=1)

    # take top 150
    #top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
    #    "feature"].values.tolist()
    #final = final[top_list]

    final = drop_corolated_all(final)

    final["pred"] = labels
    final = final.drop(final[(final.pred == 'other')].index)
    le = LabelEncoder().fit(
                np.unique(final['pred'].unique().tolist()))

        #At the end 0 will be used for dropped values
    final['pred'] = le.transform(final['pred'])
    final['pred'] = np.abs(final['pred']-1.0)

    final = drop_corolated_target(final)
    #final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",
    #             index=False)
    final.to_csv(init_data_path+ "/after_preprocessing/HATE_after_preprocessing_no_glove.csv",index=False)

# ___________________________________________________ HCDR - AMIT ________________________________________________________

def preprocess_credit(init_data_path):
    # Taken from https://www.kaggle.com/code/lolokiller/model-with-bayesian-optimization
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    try: # second preprocess if needed
        df = pd.read_csv(init_data_path+ "/after_preprocessing/HCDR_after_preprocessing_.csv")
        preds = df['pred']
        top_features = np.array(pd.read_csv(init_data_path + "/top_importance.csv")).flatten()
        top_features = np.sort(top_features)
        #features = x_train_target.columns.to_frame()
       
        new_df = df.iloc[:,top_features]
        new_df['pred']= preds
        new_df.to_csv(init_data_path+ "/after_preprocessing/HCDR_after_preprocessing_top_112.csv",index=False)
        
        x = new_df.copy()
        x = drop_corolated_target(x)
        final = x.drop(['pred'], axis=1)
        final = drop_corolated_all(final)

        features = final.columns.to_frame()
        features.to_csv(init_data_path +'/edittible_features.csv', index=False)
        
        final['pred']= preds
        final.to_csv(init_data_path+ "/after_preprocessing/HCDR_after_preprocessing_top_.csv",index=False)
        
        
        return

    except:
        pass
    
    df = pd.read_csv(init_data_path + "/original/application_train.csv")

    # Remove outliers
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Feature engineering
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_group(x))
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
    df = do_median(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
    df = do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    df = do_mean(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    df = do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    df = do_mean(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    df = do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    df = do_mean(df, group, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
    df = do_mean(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
    df = do_std(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')
    
    # Encoding categorical
    df, le_encoded_cols = label_encoder(df, None)
    df = drop_application_columns(df)
    df = pd.get_dummies(df)

    # Add Bureau data
    bureau = pd.read_csv(init_data_path + "/original/bureau.csv")
    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
    bureau, categorical_cols = one_hot_encoder(bureau, nan_as_category=False)
    #bureau = bureau.merge(get_bureau_balance(init_data_path+'/original'), how='left', on='SK_ID_BUREAU')
    #bureau['STATUS_12345'] = 0
    #for i in range(1, 6):
    #    bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]

    features = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM',
                'AMT_CREDIT_SUM_DEBT', 'DEBT_PERCENTAGE', 'DEBT_CREDIT_DIFF', 'STATUS_0', 'STATUS_12345']
    #agg_length = bureau.groupby('MONTHS_BALANCE_SIZE')[features].mean().reset_index()
    #agg_length.rename({feat: 'LL_' + feat for feat in features}, axis=1, inplace=True)
    #bureau = bureau.merge(agg_length, how='left', on='MONTHS_BALANCE_SIZE')
    BUREAU_AGG = {
        'SK_ID_BUREAU': ['nunique'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean', 'sum'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'STATUS_0': ['mean'],
        'STATUS_1': ['mean'],
        'STATUS_12345': ['mean'],
        'STATUS_C': ['mean'],
        'STATUS_X': ['mean'],
        'CREDIT_ACTIVE_Active': ['mean'],
        'CREDIT_ACTIVE_Closed': ['mean'],
        'CREDIT_ACTIVE_Sold': ['mean'],
        'CREDIT_TYPE_Consumer credit': ['mean'],
        'CREDIT_TYPE_Credit card': ['mean'],
        'CREDIT_TYPE_Car loan': ['mean'],
        'CREDIT_TYPE_Mortgage': ['mean'],
        'CREDIT_TYPE_Microloan': ['mean'],
        'LL_AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'LL_DEBT_CREDIT_DIFF': ['mean'],
        'LL_STATUS_12345': ['mean'],
    }
    BUREAU_ACTIVE_AGG = {
        'DAYS_CREDIT': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'mean'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'CREDIT_TO_ANNUITY_RATIO': ['mean'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    }
    BUREAU_CLOSED_AGG = {
        'DAYS_CREDIT': ['max', 'var'],
        'DAYS_CREDIT_ENDDATE': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
        'DAYS_CREDIT_UPDATE': ['max'],
        'ENDDATE_DIF': ['mean'],
        'STATUS_12345': ['mean'],
    }
    BUREAU_LOAN_TYPE_AGG = {
        'DAYS_CREDIT': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'DAYS_CREDIT_ENDDATE': ['max'],
    }
    BUREAU_TIME_AGG = {
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'STATUS_0': ['mean'],
        'STATUS_12345': ['mean'],
    }
    '''
    agg_bureau = group_f(bureau, 'BUREAU_', BUREAU_AGG)
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    agg_bureau = group_and_merge(active, agg_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    agg_bureau = group_and_merge(closed, agg_bureau, 'BUREAU_CLOSED_', BUREAU_CLOSED_AGG)
    for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
        type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
        prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
        agg_bureau = group_and_merge(type_df, agg_bureau, prefix, BUREAU_LOAN_TYPE_AGG)
    for time_frame in [6, 12]:
        prefix = "BUREAU_LAST{}M_".format(time_frame)
        time_frame_df = bureau[bureau['DAYS_CREDIT'] >= -30 * time_frame]
        agg_bureau = group_and_merge(time_frame_df, agg_bureau, prefix, BUREAU_TIME_AGG)
    sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])
    gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()
    gr.rename({'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_LAST_LOAN_MAX_OVERDUE'}, inplace=True)
    agg_bureau = agg_bureau.merge(gr, on='SK_ID_CURR', how='left')
    agg_bureau['BUREAU_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
    agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
    df = pd.merge(df, agg_bureau, on='SK_ID_CURR', how='left')
    '''
    # Adding Installments/ previous application data
    prev = pd.read_csv(os.path.join(init_data_path, 'original/previous_application.csv'))
    pay = pd.read_csv(os.path.join(init_data_path, 'original/installments_payments.csv'))
    PREVIOUS_AGG = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_TERMINATION': ['max'],
        # Engineered features
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
        'DOWN_PAYMENT_TO_CREDIT': ['mean'],
    }
    PREVIOUS_ACTIVE_AGG = {
        'SK_ID_PREV': ['nunique'],
        'SIMPLE_INTERESTS': ['mean'],
        'AMT_ANNUITY': ['max', 'sum'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['sum'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'AMT_PAYMENT': ['sum'],
        'INSTALMENT_PAYMENT_DIFF': ['mean', 'max'],
        'REMAINING_DEBT': ['max', 'mean', 'sum'],
        'REPAYMENT_RATIO': ['mean'],
    }
    PREVIOUS_LATE_PAYMENTS_AGG = {
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    PREVIOUS_LOAN_TYPE_AGG = {
        'AMT_CREDIT': ['sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'SIMPLE_INTERESTS': ['min', 'mean', 'max', 'var'],
        'APPLICATION_CREDIT_DIFF': ['min', 'var'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['max'],
        'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean'],
    }
    PREVIOUS_TIME_AGG = {
        'AMT_CREDIT': ['sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'SIMPLE_INTERESTS': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    PREVIOUS_APPROVED_AGG = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max'],
        'AMT_GOODS_PRICE': ['max'],
        'HOUR_APPR_PROCESS_START': ['min', 'max'],
        'DAYS_DECISION': ['min', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_TERMINATION': ['mean'],
        # Engineered features
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'APPLICATION_CREDIT_DIFF': ['max'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        # The following features are only for approved applications
        'DAYS_FIRST_DRAWING': ['max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['max', 'mean'],
        'DAYS_LAST_DUE_DIFF': ['min', 'max', 'mean'],
        'SIMPLE_INTERESTS': ['min', 'max', 'mean'],
    }
    PREVIOUS_REFUSED_AGG = {
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['min', 'max'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
        'APPLICATION_CREDIT_RATIO': ['min', 'mean'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    ohe_columns = [
        'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
        'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
        'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']
    prev, categorical_cols = one_hot_encoder(prev, ohe_columns, nan_as_category=False)
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    active_df = approved[approved['DAYS_LAST_DUE'] == 365243]
    active_pay = pay[pay['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]
    active_pay_agg = active_pay.groupby('SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
    active_pay_agg.reset_index(inplace=True)
    active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']
    active_df = active_df.merge(active_pay_agg, on='SK_ID_PREV', how='left')
    active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']
    active_df['REPAYMENT_RATIO'] = active_df['AMT_PAYMENT'] / active_df['AMT_CREDIT']
    active_agg_df = group_f(active_df, 'PREV_ACTIVE_', PREVIOUS_ACTIVE_AGG)
    active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM'] / \
                                             active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    approved['DAYS_LAST_DUE_DIFF'] = approved['DAYS_LAST_DUE_1ST_VERSION'] - approved['DAYS_LAST_DUE']

    categorical_agg = {key: ['mean'] for key in categorical_cols}

    agg_prev = group_f(prev, 'PREV_', {**PREVIOUS_AGG, **categorical_agg})
    agg_prev = agg_prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
    agg_prev = group_and_merge(approved, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    agg_prev = group_and_merge(refused, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
    for loan_type in ['Consumer loans', 'Cash loans']:
        type_df = prev[prev['NAME_CONTRACT_TYPE_{}'.format(loan_type)] == 1]
        prefix = 'PREV_' + loan_type.split(" ")[0] + '_'
        agg_prev = group_and_merge(type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG)
    pay['LATE_PAYMENT'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
    pay['LATE_PAYMENT'] = pay['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
    dpd_id = pay[pay['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()

    agg_dpd = group_and_merge(prev[prev['SK_ID_PREV'].isin(dpd_id)], agg_prev,
                              'PREV_LATE_', PREVIOUS_LATE_PAYMENTS_AGG)
    for time_frame in [12, 24]:
        time_frame_df = prev[prev['DAYS_DECISION'] >= -30 * time_frame]
        prefix = 'PREV_LAST{}M_'.format(time_frame)
        agg_prev = group_and_merge(time_frame_df, agg_prev, prefix, PREVIOUS_TIME_AGG)
    df = pd.merge(df, agg_prev, on='SK_ID_CURR', how='left')

    # Fill NA and scale
    labels = df['TARGET']
    df = df.drop(columns=["TARGET"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    #YAEL
    #pd.set_option('display.max_rows', 500)

    NA_col_train = pd.DataFrame(df.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(df))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    df = df.drop(cols_to_drop, axis=1)

    df = df.drop('SK_ID_CURR', axis=1)
    df = df.drop('CODE_GENDER', axis=1)

    colonnes = df.columns
    feature = list(df.columns)

    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)
    df = imputer.transform(df)
    pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    final["pred"] = labels
    final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/HCDR/after_preprocessing/HCDR_after_preprocessing.csv",
                 index=False)
    '''
    # drop not important
    drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    final = final.drop(drop_list, axis=1)

    # take top 150
    top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
        "feature"].values.tolist()
    final = final[top_list]

    final["pred"] = labels
    final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/credit_after_preprocessing.csv",
                 index=False)
    '''
