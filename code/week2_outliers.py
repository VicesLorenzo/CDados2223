from globals import class_climate_prepared_df, class_health_prepared_df
from globals import save_dataset, CLIMATE_DATASET_FOLDER, HEALTH_DATASET_FOLDER
from globals import CLASSIFICATION_CLIMATE_PREPARED_FILENAME, CLASSIFICATION_HEALTH_PREPARED_FILENAME
from pandas import DataFrame
from ds_charts import get_variable_types

OUTLIER_PARAM: int = 2 # define the number of stdev to use or the IQR scale (usually 1.5)
OPTION = 'iqr'  # or 'stdev'

def determine_outlier_thresholds(summary5: DataFrame, var: str):
    if 'iqr' == OPTION:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:  # OPTION == 'stdev'
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

def outliers_treatment(org_df, dataset_folder, filename):
    numeric_vars = get_variable_types(org_df)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    summary5 = org_df.describe(include='number')
    new_df = org_df.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        new_df[var] = new_df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)
    save_dataset(new_df, dataset_folder, filename)

outliers_treatment(class_climate_prepared_df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME)
outliers_treatment(class_health_prepared_df, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME)
