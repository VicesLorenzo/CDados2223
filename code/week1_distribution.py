from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_1_FOLDER
from ds_charts import get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart, bar_chart
from matplotlib.pyplot import subplots, figure
from seaborn import distplot
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm

def outliers_boxplots(df, image_folder, filename):
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars:
        return
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def outliers_per_variable(df, image_folder, filename):
    NR_STDEV : int = 2
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars:
        return
    outliers_iqr = []
    outliers_stdev = []
    summary5 = df.describe(include='number')
    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            df[df[var] > summary5[var]['75%']  + iqr].count()[var] +
            df[df[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            df[df[var] > summary5[var]['mean'] + std].count()[var] +
            df[df[var] < summary5[var]['mean'] - std].count()[var]]
    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def histogram_numeric(df, image_folder, filename):
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars:
        return
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(df[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def histogram_numeric_trend(df, image_folder, filename):
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars:
        return
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(df[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def histogram_numeric_distribution(df, image_folder, filename):
    def compute_known_distributions(x_values):
        distributions = dict()
        # Gaussian
        mean, sigma = norm.fit(x_values)
        distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
        # LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
        return distributions
    def histogram_with_distributions(ax, series: Series, var: str):
        values = series.sort_values().values
        ax.hist(values, 20, density=True)
        distributions = compute_known_distributions(values)
        multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')
    numeric_vars = get_variable_types(df)['Numeric']
    if [] == numeric_vars:
        return
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], df[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

def histogram_symbolic(df, image_folder, filename):
    symbolic_vars = get_variable_types(df)['Symbolic']
    if [] == symbolic_vars:
        return
    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = df[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    save_image(image_folder, WEEK_1_FOLDER, filename, show_flag=False)

outliers_boxplots(class_climate_df, CLIMATE_IMAGE_FOLDER, "outliers_boxplots")
outliers_boxplots(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "outliers_boxplots_encoded")
outliers_boxplots(class_health_df, HEALTH_IMAGE_FOLDER, "outliers_boxplots")
outliers_boxplots(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "outliers_boxplots_encoded")

outliers_per_variable(class_climate_df, CLIMATE_IMAGE_FOLDER, "outliers_per_variable")
outliers_per_variable(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "outliers_per_variable_encoded")
outliers_per_variable(class_health_df, HEALTH_IMAGE_FOLDER, "outliers_per_variable")
outliers_per_variable(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "outliers_per_variable_encoded")

histogram_numeric(class_climate_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric")
histogram_numeric(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric_encoded")
histogram_numeric(class_health_df, HEALTH_IMAGE_FOLDER, "histogram_numeric")
histogram_numeric(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "histogram_numeric_encoded")

histogram_numeric_trend(class_climate_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric_trend")
histogram_numeric_trend(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric_trend_encoded")
histogram_numeric_trend(class_health_df, HEALTH_IMAGE_FOLDER, "histogram_numeric_trend")
histogram_numeric_trend(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "histogram_numeric_trend_encoded")

#todo
histogram_numeric_distribution(class_climate_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric_distribution")
histogram_numeric_distribution(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "histogram_numeric_distribution_encoded")
histogram_numeric_distribution(class_health_df, HEALTH_IMAGE_FOLDER, "histogram_numeric_distribution")
histogram_numeric_distribution(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "histogram_numeric_distribution_encoded")

histogram_symbolic(class_climate_df, CLIMATE_IMAGE_FOLDER, "histogram_symbolic")
histogram_symbolic(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "histogram_symbolic_encoded")
histogram_symbolic(class_health_df, HEALTH_IMAGE_FOLDER, "histogram_symbolic")
histogram_symbolic(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "histogram_symbolic_encoded")
