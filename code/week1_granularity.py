from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_1_FOLDER
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots

def granularity_single_bin(df, image_folder, filename):
    variables = get_variable_types(df)['Numeric']
    if [] == variables:
        return
    rows, cols = choose_grid(len(variables))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(variables)):
        axs[i, j].set_title('Histogram for %s'%variables[n])
        axs[i, j].set_xlabel(variables[n])
        axs[i, j].set_ylabel('nr records')
        axs[i, j].hist(df[variables[n]].values, bins=100)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def granularity_multiple_bin(df, image_folder, filename):
    variables = get_variable_types(df)['Numeric']
    if [] == variables:
        return
    rows = len(variables)
    bins = (10, 100, 400)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            axs[i, j].hist(df[variables[i]].values, bins=bins[j])
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

granularity_single_bin(class_climate_df, CLIMATE_IMAGE_FOLDER, "granularity_single_bin")
granularity_single_bin(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "granularity_single_bin_encoded")
granularity_single_bin(class_health_df, HEALTH_IMAGE_FOLDER, "granularity_single_bin")
granularity_single_bin(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "granularity_single_bin_encoded")

granularity_multiple_bin(class_climate_df, CLIMATE_IMAGE_FOLDER, "granularity_multiple_bin")
granularity_multiple_bin(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "granularity_multiple_bin_encoded")
granularity_multiple_bin(class_health_df, HEALTH_IMAGE_FOLDER, "granularity_multiple_bin")
granularity_multiple_bin(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "granularity_multiple_bin_encoded")
