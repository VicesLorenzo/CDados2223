from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_1_FOLDER
from matplotlib.pyplot import figure
from ds_charts import bar_chart, get_variable_types

def records_per_variable(df, image_folder, filename):
    figure(figsize=(4,2))
    values = {'nr records': df.shape[0], 'nr variables': df.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def variables_per_type(df, image_folder, filename):
    variable_types = get_variable_types(df)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def missing_values(df, image_folder, filename):
    mv = {}
    for var in df:
        nr = df[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    if not mv:
        return
    figure()
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

records_per_variable(class_climate_df, CLIMATE_IMAGE_FOLDER, "records_per_variable")
records_per_variable(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "records_per_variable_encoded")
records_per_variable(class_health_df, HEALTH_IMAGE_FOLDER, "records_per_variable")
records_per_variable(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "records_per_variable_encoded")

variables_per_type(class_climate_df, CLIMATE_IMAGE_FOLDER, "variables_per_type")
variables_per_type(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "variables_per_type_encoded")
variables_per_type(class_health_df, HEALTH_IMAGE_FOLDER, "variables_per_type")
variables_per_type(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "variables_per_type_encoded")

missing_values(class_climate_df, CLIMATE_IMAGE_FOLDER, "missing_values")
missing_values(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "missing_values_encoded")
missing_values(class_health_df, HEALTH_IMAGE_FOLDER, "missing_values")
missing_values(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "missing_values_encoded")
