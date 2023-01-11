from globals import forecast_health_prepared_df, forecast_climate_prepared_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from pandas import DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from globals import save_image, HEALTH_IMAGE_FOLDER, WEEK_5_FOLDER
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

    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    data.boxplot(ax=axs[0, 0])
    axs[0, 1].set_title('Z-score normalization')
    norm_data_zscore.boxplot(ax=axs[0, 1])
    axs[0, 2].set_title('MinMax normalization')
    norm_data_minmax.boxplot(ax=axs[0, 2])


    save_image(image_folder, WEEK_5_FOLDER, filename, show_flag=False)

#scaling não vale a pena ser aplicado a nenhum dos datasets

scaling(forecast_climate_prepared_df, CLIMATE_IMAGE_FOLDER, "forecast_scaling_prepared")
scaling(forecast_health_prepared_df, HEALTH_IMAGE_FOLDER, "forecast_scaling_prepared")