from globals import class_climate_prepared_df, class_health_prepared_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from pandas import DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from globals import save_image, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
from pandas import DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from week2_study import nb_study, knn_study
from ds_charts import get_variable_types

register_matplotlib_converters()

def scaling(data, image_folder, filename):
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    def zscore_scaling(df_nr, df_sb, df_bool):
        transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
        tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
        norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
        return norm_data_zscore

    def minmax_scaling(df_nr, df_sb, df_bool):
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
        norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
        return norm_data_minmax

    norm_data_zscore = zscore_scaling(df_nr, df_sb, df_bool)
    norm_data_minmax = minmax_scaling(df_nr, df_sb, df_bool)

    #if(image_folder == HEALTH_IMAGE_FOLDER):
        #nb_study(norm_data_zscore, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "nb_zcore_scaling")
        #nb_study(norm_data_minmax, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "nb_minmax_scaling")
        #knn_study(norm_data_zscore, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "knn_zcore_scaling")
        #knn_study(norm_data_minmax, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "knn_minmax_scaling")
    if(image_folder == CLIMATE_IMAGE_FOLDER):
        nb_study(norm_data_zscore, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "nb_zcore_scaling")
        nb_study(norm_data_minmax, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "nb_minmax_scaling")
        knn_study(norm_data_zscore, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "knn_zcore_scaling")
        knn_study(norm_data_minmax, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "knn_minmax_scaling")

    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    data.boxplot(ax=axs[0, 0])
    axs[0, 1].set_title('Z-score normalization')
    norm_data_zscore.boxplot(ax=axs[0, 1])
    axs[0, 2].set_title('MinMax normalization')
    norm_data_minmax.boxplot(ax=axs[0, 2])
    save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)

scaling(class_climate_prepared_df, CLIMATE_IMAGE_FOLDER, "scaling_encoded")
#scaling(class_health_prepared_df, HEALTH_IMAGE_FOLDER, "scaling_encoded")