from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from ds_charts import subplots
from globals import save_image, HEALTH_IMAGE_FOLDER, CLIMATE_IMAGE_FOLDER, forecast_climate_df, forecast_health_df, WEEK_5_FOLDER
from numpy import ones
from pandas import Series

def dimensionality(data, image_folder, filename):
    print(data.shape[0])
    figure(figsize=(3*HEIGHT, HEIGHT))
    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(data, x_label='Date', y_label='glucose', title='Health')
    else:
        plot_series(data, x_label='date', y_label='QV2M', title='Climate')
    save_image(image_folder, WEEK_5_FOLDER, filename, show_flag=False)

def granularity(data, image_folder, filename, filename2, filename3, filename4):
    day_df = data.copy().groupby(data.index.date).mean()
    figure(figsize=(3*HEIGHT, HEIGHT))
    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(day_df, x_label='Date', y_label='glucose', title='Daily glucose')
    else:
        plot_series(day_df, x_label='date', y_label='QV2M', title='Daily drought')
    xticks(rotation = 45)
    save_image(image_folder, WEEK_5_FOLDER, filename, show_flag=False)

    index = data.index.to_period('W')
    week_df = data.copy().groupby(index).mean()
    week_df['date'] = index.drop_duplicates().to_timestamp()
    week_df.set_index('date', drop=True, inplace=True)
    figure(figsize=(3*HEIGHT, HEIGHT))
    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(week_df, x_label='Date', y_label='glucose', title='Weekly glucose')
    else:
        plot_series(week_df, x_label='date', y_label='QV2M', title='Weekly drought')
    xticks(rotation = 45)
    save_image(image_folder, WEEK_5_FOLDER, filename2, show_flag=False)

    index2 = data.index.to_period('M')
    month_df = data.copy().groupby(index2).mean()
    month_df['date'] = index2.drop_duplicates().to_timestamp()
    month_df.set_index('date', drop=True, inplace=True)
    figure(figsize=(3*HEIGHT, HEIGHT))
    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(month_df, x_label='Date', y_label='glucose', title='Monthly glucose')
    else:
        plot_series(month_df, x_label='date', y_label='QV2M', title='Monthly drought')
    save_image(image_folder, WEEK_5_FOLDER, filename3, show_flag=False)

    index3 = data.index.to_period('Q')
    quarter_df = data.copy().groupby(index3).mean()
    quarter_df['date'] = index3.drop_duplicates().to_timestamp()
    quarter_df.set_index('date', drop=True, inplace=True)
    figure(figsize=(3*HEIGHT, HEIGHT))
    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(quarter_df, x_label='Date', y_label='glucose', title='Quarterly glucose')
    else:
        plot_series(quarter_df, x_label='date', y_label='QV2M', title='Quarterly drought')
    save_image(image_folder, WEEK_5_FOLDER, filename4, show_flag=False)

def distribution(data, image_folder, filename, filename2, filename3):
    index = data.index.to_period('W')
    week_df = data.copy().groupby(index).mean()
    week_df['date'] = index.drop_duplicates().to_timestamp()
    week_df.set_index('date', drop=True, inplace=True)
    _, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
    axs[0].grid(False)
    axs[0].set_axis_off()
    axs[0].set_title('HOURLY', fontweight="bold")
    axs[0].text(0, 0, str(data.describe()))
    axs[1].grid(False)
    axs[1].set_axis_off()
    axs[1].set_title('WEEKLY', fontweight="bold")
    axs[1].text(0, 0, str(week_df.describe()))
    
    _, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
    data.boxplot(ax=axs[0])
    week_df.boxplot(ax=axs[1])
    save_image(image_folder, WEEK_5_FOLDER, filename, show_flag=False)

    bins = (10, 25, 50)
    _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
    for j in range(len(bins)):
        axs[j].set_title('Histogram for hourly meter_reading %d bins'%bins[j])
        if(image_folder == HEALTH_IMAGE_FOLDER):
            axs[j].set_xlabel('glucose')
        else:
            axs[j].set_xlabel('QV2M')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data.values, bins=bins[j])
    save_image(image_folder, WEEK_5_FOLDER, filename2, show_flag=False)


    _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
    for j in range(len(bins)):
        axs[j].set_title('Histogram for weekly meter_reading %d bins'%bins[j])
        if(image_folder == HEALTH_IMAGE_FOLDER):
            axs[j].set_xlabel('glucose')
        else:
            axs[j].set_xlabel('QV2M')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(week_df.values, bins=bins[j])
    save_image(image_folder, WEEK_5_FOLDER, filename3, show_flag=False)

def stationarity(data, image_folder, filename):

    if(image_folder == HEALTH_IMAGE_FOLDER):
        dt_series = Series(data['Glucose'])
    else:
        dt_series = Series(data['QV2M'])
    mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
    
    
    if(image_folder == HEALTH_IMAGE_FOLDER):
        series = {'Glucose': dt_series, 'mean': mean_line}
    else:
        series = {'QV2M': dt_series, 'mean': mean_line}
    figure(figsize=(3*HEIGHT, HEIGHT))


    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_series(series, x_label='Date', y_label='glucose', title='Stationary Study', show_std=True)
    else:
        plot_series(series, x_label='date', y_label='QV2M', title='Stationary Study', show_std=True)
    save_image(image_folder, WEEK_5_FOLDER, filename, show_flag=False)


#stationarity(forecast_climate_df, CLIMATE_IMAGE_FOLDER, "forecast_stationarity")
#stationarity(forecast_health_df, HEALTH_IMAGE_FOLDER, "forecast_stationarity")

#distribution(forecast_climate_df, CLIMATE_IMAGE_FOLDER, "forecast_distribution", "forecast_distribution_histogram_hourly", "forecast_distribution_histogram_weekly")
#distribution(forecast_health_df, HEALTH_IMAGE_FOLDER, "forecast_distribution", "forecast_distribution_histogram_hourly", "forecast_distribution_histogram_weekly")


#dimensionality(forecast_climate_df, CLIMATE_IMAGE_FOLDER, "forecast_dimensionality")
#dimensionality(forecast_health_df, HEALTH_IMAGE_FOLDER, "forecast_dimensionality")

#granularity(forecast_climate_df, CLIMATE_IMAGE_FOLDER, "forecast_granularity_daily", "forecast_granularity_weekly", "forecast_granularity_monthly",
#"forecast_granularity_quarterly")
#granularity(forecast_health_df, HEALTH_IMAGE_FOLDER, "forecast_granularity_daily", "forecast_granularity_weekly", "forecast_granularity_monthly",
#"forecast_granularity_quarterly")

