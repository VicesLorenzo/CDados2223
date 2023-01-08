from globals import class_climate_prepared_df, class_health_prepared_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_3_FOLDER
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import save_dataset, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME
from ds_charts import get_variable_types, bar_chart
from pandas import DataFrame
from matplotlib.pyplot import figure, title
from seaborn import heatmap
from week2_study import nb_study, knn_study

CORRELATION_THRESHOLD = 0.9
VARIANCE_THRESHOLD = 0.1

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
    #print('Variables to drop', sel_2drop)
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

nb_study(class_climate_prepared_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "simple_feature_engineering_nb_study")
knn_study(class_climate_prepared_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "simple_feature_engineering_knn_study")

climate_redundant_2drop, climate_corr_mtx = select_redundant(class_climate_prepared_df.corr(), CORRELATION_THRESHOLD)
plot_heatmap(climate_corr_mtx, CLIMATE_IMAGE_FOLDER, "correlation_study_encoded")
climate_dropped_df = drop_redundant(class_climate_prepared_df, climate_redundant_2drop)

nb_study(climate_dropped_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "dropped_redundant_feature_engineering_nb_study")
knn_study(climate_dropped_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "dropped_redundant_engineering_knn_study")

climate_variance_2drop = select_low_variance(climate_dropped_df[get_variable_types(climate_dropped_df)['Numeric']], VARIANCE_THRESHOLD, CLIMATE_IMAGE_FOLDER, "variance_study_encoded")

# No dataset climate dropamos as colunas com correlacao maior que 0.9 e nao dropamos por variance pq iriamos perder muitas colunas
#save_dataset(climate_dropped_df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME)

nb_study(class_health_prepared_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "simple_feature_engineering_nb_study")
knn_study(class_health_prepared_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "simple_feature_engineering_knn_study")


health_reduntant_2drop, health_corr_mtx = select_redundant(class_health_prepared_df.corr(), CORRELATION_THRESHOLD)
plot_heatmap(health_corr_mtx, HEALTH_IMAGE_FOLDER, "correlation_study_encoded")
health_dropped_df = drop_redundant(class_health_prepared_df, health_reduntant_2drop)

nb_study(health_dropped_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "dropped_redundant_engineering_nb_study")
knn_study(health_dropped_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "dropped_redundant_engineering_knn_study")

health_variance_2drop = select_low_variance(health_dropped_df[get_variable_types(health_dropped_df)['Numeric']], VARIANCE_THRESHOLD, HEALTH_IMAGE_FOLDER, "variance_study_encoded")

# No dataset health não tem correlation entre variáveis portanto não dropamos nenhuma e nao dropamos por variance pq iriamos perder muitas colunas
#save_dataset(class_health_prepared_df, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME)
