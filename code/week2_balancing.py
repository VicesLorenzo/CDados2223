from globals import class_health_prepared_df, class_climate_prepared_df
from globals import CLASSIFICATION_HEALTH_TARGET, CLASSIFICATION_CLIMATE_TARGET
from globals import save_image, WEEK_2_FOLDER, HEALTH_IMAGE_FOLDER, CLIMATE_IMAGE_FOLDER
from matplotlib.pyplot import figure
from ds_charts import bar_chart

def class_balance(df, class_var, image_folder, filename):
    target_count = df[class_var].value_counts().sort_index()
    figure()
    bar_chart(target_count.index.map(str), target_count.values, title='Class balance')
    save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)

class_balance(class_health_prepared_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "class_balance_per_value")
class_balance(class_climate_prepared_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "class_balance_per_value")
