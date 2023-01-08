from globals import class_climate_prepared_df, class_health_prepared_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import CLASSIFICATION_CLIMATE_PREPARED_TRAIN_FILENAME, CLASSIFICATION_CLIMATE_PREPARED_TEST_FILENAME
from globals import CLASSIFICATION_HEALTH_PREPARED_TRAIN_FILENAME, CLASSIFICATION_HEALTH_PREPARED_TEST_FILENAME
from globals import save_dataset, CLIMATE_DATASET_FOLDER, HEALTH_DATASET_FOLDER
from sklearn.model_selection import train_test_split
from pandas import unique, DataFrame
import numpy as np
    
def split(df, target_class):
    data = df.copy(deep=True)
    y = data.pop(target_class).values
    X = data.values
    labels = unique(y)
    labels.sort()
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    train = np.column_stack([trnX, trnY])
    train_df = DataFrame(train, columns = df.columns)
    test = np.column_stack([tstX, tstY])
    test_df = DataFrame(test, columns = df.columns)
    return train_df, test_df

climate_train_df, climate_test_df = split(class_climate_prepared_df, CLASSIFICATION_CLIMATE_TARGET)
save_dataset(climate_train_df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_TRAIN_FILENAME)
save_dataset(climate_test_df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_TEST_FILENAME)

health_train_df, health_test_df = split(class_health_prepared_df, CLASSIFICATION_HEALTH_TARGET)
save_dataset(health_train_df, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_TRAIN_FILENAME)
save_dataset(health_test_df, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_TEST_FILENAME)
