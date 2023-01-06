from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_3_FOLDER
from ds_charts import get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart, bar_chart
from matplotlib.pyplot import subplots, figure
from seaborn import distplot
from numpy import log
from scipy.stats import norm, expon, lognorm
from pandas import DataFrame
from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap

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
    return corr_mtx

def plot_heatmap(corr_mtx, image_folder, filename):

    if corr_mtx.empty:
        raise ValueError('Matrix is empty.')

    figure(figsize=[10, 10])
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
    title('Filtered Correlation Analysis')
    save_image(image_folder, WEEK_3_FOLDER, filename, show_flag=False)


plot_heatmap(select_redundant(class_climate_encoded_df.corr(), THRESHOLD), CLIMATE_IMAGE_FOLDER, "correlation_study_encoded")
plot_heatmap(select_redundant(class_health_encoded_df.corr(), THRESHOLD), HEALTH_IMAGE_FOLDER, "correlation_study_encoded")