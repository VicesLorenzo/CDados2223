from globals import class_health_prepared_train_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_climate_prepared_test_df
from globals import CLASSIFICATION_HEALTH_TARGET, CLASSIFICATION_CLIMATE_TARGET
from globals import save_image, WEEK_3_FOLDER, HEALTH_IMAGE_FOLDER, CLIMATE_IMAGE_FOLDER
from matplotlib.pyplot import figure
from ds_charts import bar_chart
from pandas import DataFrame, concat
import numpy as np

def class_balance(df, class_var, image_folder, filename):
    target_count = df[class_var].value_counts().sort_index()
    figure()
    bar_chart(target_count.index.map(str), target_count.values, title='Class balance')
    save_image(image_folder, WEEK_3_FOLDER, filename, show_flag=False)

def binary_undersampling(original, class_var):
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    return df_under

def ternary_undersampling(original, class_var):
    values = original[class_var].unique().tolist()
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    values.remove(positive_class)
    values.remove(negative_class)
    remain_class = values[0]
    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_remains = original[original[class_var] == remain_class]
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_remain_sample = DataFrame(df_remains.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample, df_remain_sample], axis=0)
    return df_under

def binary_oversampling(original, class_var):
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    return df_over

def ternary_oversampling(original, class_var):
    values = original[class_var].unique().tolist()
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    values.remove(positive_class)
    values.remove(negative_class)
    remain_class = values[0]
    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_remains = original[original[class_var] == remain_class]
    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_remain_sample = DataFrame(df_remains.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives, df_remain_sample], axis=0)
    return df_over

class_balance(class_health_prepared_train_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "class_balance_train")
class_balance(class_climate_prepared_train_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "class_balance_train")

# Como no dataset climate o class balance está fixe, nao vamos fazer balancing desse dataset

climate_under_df = binary_undersampling(class_climate_prepared_train_df, CLASSIFICATION_CLIMATE_TARGET)
class_balance(climate_under_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "class_balance_train_under")
climate_over_df = binary_oversampling(class_climate_prepared_train_df, CLASSIFICATION_CLIMATE_TARGET)
class_balance(climate_over_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "class_balance_train_over")

health_under_df = ternary_undersampling(class_health_prepared_train_df, CLASSIFICATION_HEALTH_TARGET)
class_balance(health_under_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "class_balance_train_under")
health_over_df = ternary_oversampling(class_health_prepared_train_df, CLASSIFICATION_HEALTH_TARGET)
class_balance(health_over_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "class_balance_train_over")
