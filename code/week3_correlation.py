from globals import CLASSIFICATION_CLIMATE_PREPARED_FILENAME, CLASSIFICATION_HEALTH_PREPARED_FILENAME, CLIMATE_DATASET_FOLDER, HEALTH_DATASET_FOLDER, class_climate_encoded_df, class_health_encoded_df, save_dataset
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_3_FOLDER
from ds_charts import get_variable_types, bar_chart
from pandas import DataFrame
from matplotlib.pyplot import figure, title
from seaborn import heatmap
from globals import class_climate_prepared_df, class_health_prepared_df

THRESHOLD = 0.9

def select_redundant(corr_mtx, threshold: float) -> tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

def plot_heatmap(corr_mtx, image_folder, filename):

    if corr_mtx.empty:
        raise ValueError('Matrix is empty.')

    figure(figsize=[10, 10])
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
    title('Filtered Correlation Analysis')
    save_image(image_folder, WEEK_3_FOLDER, filename, show_flag=False)

def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df

def select_low_variance(data: DataFrame, threshold: float, image_folder, filename) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    figure(figsize=[20, 10])
    bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
    save_image(image_folder, WEEK_3_FOLDER, filename, show_flag=False)
    return lst_variables

drop, corr_mtx = select_redundant(class_climate_prepared_df.corr(), THRESHOLD)
plot_heatmap(corr_mtx, CLIMATE_IMAGE_FOLDER, "correlation_study_encoded")
df = drop_redundant(class_climate_prepared_df, drop)

save_dataset(df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME)

numeric = get_variable_types(df)['Numeric']
vars_2drop = select_low_variance(df[numeric], 0.1, CLIMATE_IMAGE_FOLDER, "variance_study_encoded")

#Health Dataset não tem correlation entre variáveis portanto não dropamos nenhuma

drop2, corr_mtx2 = select_redundant(class_health_prepared_df.corr(), THRESHOLD)
plot_heatmap(corr_mtx2, HEALTH_IMAGE_FOLDER, "correlation_study_encoded")
dg = drop_redundant(corr_mtx2, drop2)

save_dataset(dg, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME)