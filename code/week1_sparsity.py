from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_1_FOLDER
from ds_charts import get_variable_types, HEIGHT
import matplotlib
from matplotlib.pyplot import subplots, title, figure
from seaborn import heatmap

def sparsity_numeric(df, image_folder, filename):
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars or len(numeric_vars) == 1:
        return
    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(df[var1], df[var2])
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def sparsity_symbolic(df, image_folder, filename):
    symbolic_vars = get_variable_types(df)['Symbolic']
    if [] == symbolic_vars or len(symbolic_vars) == 1:
        return
    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(df[var1], df[var2])
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def correlation_analysis(df, image_folder, filename):
    corr_mtx = abs(df.corr())
    fig = figure(figsize=[12, 12])
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

matplotlib.use('Agg')

#sparsity_numeric(class_climate_df, CLIMATE_IMAGE_FOLDER, "sparsity_numeric")
sparsity_numeric(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "sparsity_numeric_encoded")
#sparsity_numeric(class_health_df, HEALTH_IMAGE_FOLDER, "sparsity_numeric")
sparsity_numeric(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "sparsity_numeric_encoded")

#sparsity_symbolic(class_climate_df, CLIMATE_IMAGE_FOLDER, "sparsity_symbolic")
sparsity_symbolic(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "sparsity_symbolic_encoded")
#sparsity_symbolic(class_health_df, HEALTH_IMAGE_FOLDER, "sparsity_symbolic")
sparsity_symbolic(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "sparsity_symbolic_encoded")

#correlation_analysis(class_climate_df, CLIMATE_IMAGE_FOLDER, "correlation_analysis")
correlation_analysis(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "correlation_analysis_encoded")
#correlation_analysis(class_health_df, HEALTH_IMAGE_FOLDER, "correlation_analysis")
correlation_analysis(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "correlation_analysis_encoded")
