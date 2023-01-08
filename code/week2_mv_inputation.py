from globals import class_health_encoded_df, class_climate_encoded_df
from globals import CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER
from globals import save_dataset, HEALTH_DATASET_FOLDER, CLIMATE_DATASET_FOLDER
from globals import CLASSIFICATION_CLIMATE_PREPARED_FILENAME, CLASSIFICATION_HEALTH_PREPARED_FILENAME
from week2_study import nb_study, knn_study
from pandas import DataFrame, concat
from ds_charts import get_variable_types
from numpy import nan
from sklearn.impute import SimpleImputer

def missing_values(df):
    mv = {}
    for var in df:
        nr = df[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    if not mv:
        return
    return mv

def drop_columns(mv, data):
    # defines the number of records to discard entire columns
    threshold = data.shape[0] * 0.37
    missings = [c for c in mv.keys() if mv[c]>threshold]
    data = data.drop(columns=missings, inplace=False)
    return data

def value_imputation_mean(data):
    tmp_nr, tmp_bool = None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    binary_vars = variables['Binary']
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)
    df = concat([tmp_nr, tmp_bool], axis=1)
    df = df.reindex(columns=data.columns)
    return df

def value_imputation_most_frequent(data):
    tmp_nr, tmp_bool = None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    binary_vars = variables['Binary']
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)
    df = concat([tmp_nr, tmp_bool], axis=1)
    df = df.reindex(columns=data.columns)
    return df

dropped_df = drop_columns(missing_values(class_health_encoded_df), class_health_encoded_df)
climate_mean_df = value_imputation_mean(dropped_df)
#nb_study(climate_mean_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "missing_values_imputation_nb_study_mean")
#knn_study(climate_mean_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "missing_values_imputation_knn_study_mean")

climate_most_frequent_df = value_imputation_most_frequent(dropped_df)
#nb_study(climate_most_frequent_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "missing_values_imputation_nb_study_most_frequent")
#knn_study(climate_most_frequent_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "missing_values_imputation_knn_study_most_frequent")

# os resultados de ambas as alternativas foram muito semelhantes, mas o mean deu melhor resultado na naive bayes, logo optamos pelo mean
save_dataset(climate_mean_df, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME)

# o dataset climate nao tem missing values logo fica como está
save_dataset(class_climate_encoded_df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME)
