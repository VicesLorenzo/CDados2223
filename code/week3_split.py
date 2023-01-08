from sklearn.model_selection import train_test_split
from pandas import unique
import numpy as np
from globals import CLASSIFICATION_CLIMATE_PREPARED_FILENAME, CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_PREPARED_FILENAME, CLASSIFICATION_HEALTH_TARGET, CLIMATE_DATASET_FOLDER, HEALTH_DATASET_FOLDER, class_climate_prepared_df, class_health_prepared_df, save_dataset
    
def split(df, target_class):
    data = df.copy(deep=True)
    y = data.pop(target_class).values
    X = data.values
    labels = unique(y)
    labels.sort()
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    train = np.column_stack([trnX, trnY])
    test = np.column_stack([tstX, tstY])

    return train, test

df_train, df_test = split(class_climate_prepared_df, CLASSIFICATION_CLIMATE_TARGET)
#save_dataset(df, CLIMATE_DATASET_FOLDER, CLASSIFICATION_CLIMATE_PREPARED_FILENAME)


#health_prepared_dataset está vazio (provavelmente por ter corrido duas vezes o correlation? not sure mas too tired portanto vai tudo
#assim depois é só apagar os prepareds e fazer de novo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)

dg_train, dg_test = split(class_health_prepared_df, CLASSIFICATION_HEALTH_TARGET)
#save_dataset(dg, HEALTH_DATASET_FOLDER, CLASSIFICATION_HEALTH_PREPARED_FILENAME)