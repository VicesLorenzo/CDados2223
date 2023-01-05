import numpy as np
from globals import class_climate_df, class_health_df
from globals import save_dataset, CLIMATE_DATASET_FOLDER, HEALTH_DATASET_FOLDER

MISSING_VALUE = np.nan

def encode_category(df, column_name, missing_value=None):
    classes = df[column_name].unique().tolist()
    classes.remove(missing_value)
    values = [i for i in range(len(classes))]
    df[column_name] = df[column_name].replace([missing_value], MISSING_VALUE)
    df[column_name] = df[column_name].replace(classes, values)
    return df

# encoding climate dataset
class_climate_df.columns = class_climate_df.columns.str.lower()
class_climate_df["date"] = class_climate_df["date"].values.astype(np.int64) // 10 ** 9
save_dataset(class_climate_df, CLIMATE_DATASET_FOLDER, "classification_climate_encoded")

# encoding health dataset
class_health_df.columns = class_health_df.columns.str.lower()
class_health_df["race"] = class_health_df["race"].replace(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], [0, 1, 2, 3, 4, MISSING_VALUE])
class_health_df["gender"] = class_health_df["gender"].replace(["Female", "Male", "Unknown/Invalid"], [0, 1, 2])
class_health_df["age"] = class_health_df["age"].replace(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_health_df["weight"] = class_health_df["weight"].replace(["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", ">200", "?"], [0, 1, 2, 3, 4, 5, 6, 7, 8, MISSING_VALUE])
class_health_df = encode_category(class_health_df, "payer_code", missing_value="?")
class_health_df = encode_category(class_health_df, "medical_specialty", missing_value="?")
class_health_df["max_glu_serum"] = class_health_df["max_glu_serum"].replace(["None", "Norm", ">200", ">300"], [0, 1, 2, 3])
class_health_df["a1cresult"] = class_health_df["a1cresult"].replace(["None", "Norm", ">7", ">8"], [0, 1, 2, 3])
class_health_df = class_health_df.replace(to_replace=["No", "Down", "Steady", "Up"], value=[0, 1, 2, 3])
class_health_df["change"] = class_health_df["change"].replace(["No", "Ch"], [0, 1])
class_health_df["diabetesmed"] = class_health_df["diabetesmed"].replace(["No", "Yes"], [0, 1])
class_health_df["readmitted"] = class_health_df["readmitted"].replace(["NO", "<30", ">30"], [0, 1, 2])
class_health_df = encode_category(class_health_df, "diag_1", missing_value="?")
class_health_df = encode_category(class_health_df, "diag_2", missing_value="?")
class_health_df = encode_category(class_health_df, "diag_3", missing_value="?")
save_dataset(class_health_df, HEALTH_DATASET_FOLDER, "classification_health_encoded")
