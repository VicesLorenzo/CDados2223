from globals import FORECASTING_HEALTH_PREPARED_FILENAME, forecast_climate_df, forecast_health_df, FORECASTING_CLIMATE_PREPARED_FILENAME
from globals import save_dataset, HEALTH_DATASET_FOLDER, CLIMATE_DATASET_FOLDER
from week2_study import nb_study, knn_study
from pandas import DataFrame, concat
from numpy import nan
from ds_charts import get_variable_types
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
    threshold = data.shape[0] * 0.80
    missings = [c for c in mv.keys() if mv[c]>threshold]
    data = data.drop(columns=missings, inplace=False)
    return data

def value_imputation_mean(data):
    tmp_nr, tmp_bool = None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
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

#forecast_climate_mean_df = value_imputation_mean(mv_climate_df)

#forecast_climate_most_frequent_df = value_imputation_most_frequent(mv_climate_df)

mv_health_df = drop_columns(missing_values(forecast_health_df), forecast_health_df)
forecast_health_mean_df = value_imputation_mean(mv_health_df)

save_dataset(forecast_health_mean_df, HEALTH_DATASET_FOLDER, FORECASTING_HEALTH_PREPARED_FILENAME)

# os resultados de ambas as alternativas foram muito semelhantes, mas o mean deu melhor resultado na naive bayes, logo optamos pelo mean


#forecast_health_most_frequent_df = value_imputation_most_frequent(mv_health_df)

# o dataset climate nao tem missing values logo fica como está
save_dataset(forecast_climate_df, CLIMATE_DATASET_FOLDER, FORECASTING_CLIMATE_PREPARED_FILENAME)
