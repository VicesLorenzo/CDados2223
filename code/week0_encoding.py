import numpy as np
from globals import global_climate_df, global_health_df
from globals import global_save_dataset

MISSING_VALUE = np.nan

def replace_values(df, column_name, missing_value=None):
    classes = df[column_name].unique().tolist()
    classes.remove(missing_value)
    values = [i for i in range(len(classes))]
    df[column_name] = df[column_name].replace([missing_value], MISSING_VALUE)
    df[column_name] = df[column_name].replace(classes, values)
    return df

# encoding climate dataset
global_climate_df.columns = global_climate_df.columns.str.lower()
global_save_dataset(global_climate_df, "classification_climate_encoded")

# encoding health dataset
global_health_df.columns = global_health_df.columns.str.lower()
global_health_df["race"] = global_health_df["race"].replace(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], 
                                                            [0, 1, 2, 3, 4, MISSING_VALUE])
global_health_df["gender"] = global_health_df["gender"].replace(["Female", "Male", "Unknown/Invalid"], 
                                                                [0, 1, 2])
global_health_df["age"] = global_health_df["age"].replace(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], 
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
global_health_df["weight"] = global_health_df["weight"].replace(["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", ">200", "?"], 
                                                                [0, 1, 2, 3, 4, 5, 6, 7, 8, MISSING_VALUE])
global_health_df = replace_values(global_health_df, "payer_code", "?")
global_health_df = replace_values(global_health_df, "medical_specialty", "?")
global_health_df["max_glu_serum"] = global_health_df["max_glu_serum"].replace(["None", ">200", ">300", "Norm"], 
                                                                              [0, 1, 2, 3])
global_health_df["a1cresult"] = global_health_df["a1cresult"].replace(["None", ">7", ">8", "Norm"], 
                                                                      [0, 1, 2, 3])
global_health_df["metformin"] = global_health_df["metformin"].replace(["No", "Steady", "Up", "Down"], 
                                                                      [0, 1, 2, 3])
global_health_df["repaglinide"] = global_health_df["repaglinide"].replace(["No", "Steady", "Up", "Down"], 
                                                                          [0, 1, 2, 3])
global_health_df["nateglinide"] = global_health_df["nateglinide"].replace(["No", "Steady", "Up", "Down"], 
                                                                          [0, 1, 2, 3])
global_health_df["chlorpropamide"] = global_health_df["chlorpropamide"].replace(["No", "Steady", "Up", "Down"], 
                                                                                [0, 1, 2, 3])
global_health_df["glimepiride"] = global_health_df["glimepiride"].replace(["No", "Steady", "Up", "Down"], 
                                                                          [0, 1, 2, 3])
global_health_df["acetohexamide"] = global_health_df["acetohexamide"].replace(["No", "Steady"], 
                                                                              [0, 1])
global_health_df["glipizide"] = global_health_df["glipizide"].replace(["No", "Steady", "Up", "Down"], 
                                                                      [0, 1, 2, 3])
global_health_df["glyburide"] = global_health_df["glyburide"].replace(["No", "Steady", "Up", "Down"], 
                                                                      [0, 1, 2, 3])     
global_health_df["tolbutamide"] = global_health_df["tolbutamide"].replace(["No", "Steady"], 
                                                                          [0, 1])    
global_health_df["pioglitazone"] = global_health_df["pioglitazone"].replace(["No", "Steady", "Up", "Down"], 
                                                                            [0, 1, 2, 3])
global_health_df["rosiglitazone"] = global_health_df["rosiglitazone"].replace(["No", "Steady", "Up", "Down"], 
                                                                              [0, 1, 2, 3])
global_health_df["acarbose"] = global_health_df["acarbose"].replace(["No", "Steady", "Up", "Down"], 
                                                                    [0, 1, 2, 3])
global_health_df["miglitol"] = global_health_df["miglitol"].replace(["No", "Steady", "Up", "Down"], 
                                                                    [0, 1, 2, 3])                                                                    
global_health_df["troglitazone"] = global_health_df["troglitazone"].replace(["No", "Steady"], 
                                                                            [0, 1])
global_health_df["tolazamide"] = global_health_df["tolazamide"].replace(["No", "Steady", "Up"], 
                                                                        [0, 1, 2])
global_health_df["examide"] = global_health_df["examide"].replace(["No"], 
                                                                  [0])
global_health_df["citoglipton"] = global_health_df["citoglipton"].replace(["No"], 
                                                                          [0])                                                                                                                                                                                                                                                                                         
global_health_df["insulin"] = global_health_df["insulin"].replace(["No", "Steady", "Up", "Down"], 
                                                                  [0, 1, 2, 3])
global_health_df["glyburide-metformin"] = global_health_df["glyburide-metformin"].replace(["No", "Steady", "Up", "Down"], 
                                                                                          [0, 1, 2, 3])
global_health_df["glipizide-metformin"] = global_health_df["glipizide-metformin"].replace(["No", "Steady"], 
                                                                                          [0, 1])
global_health_df["glimepiride-pioglitazone"] = global_health_df["glimepiride-pioglitazone"].replace(["No", "Steady"], 
                                                                                                    [0, 1])                                                                                                                                                                                                                                                        
global_health_df["metformin-rosiglitazone"] = global_health_df["metformin-rosiglitazone"].replace(["No", "Steady"], 
                                                                                                  [0, 1])
global_health_df["metformin-pioglitazone"] = global_health_df["metformin-pioglitazone"].replace(["No", "Steady"], 
                                                                                                [0, 1])
global_health_df["change"] = global_health_df["change"].replace(["No", "Ch"], 
                                                                [0, 1])
global_health_df["diabetesmed"] = global_health_df["diabetesmed"].replace(["No", "Yes"], 
                                                                          [0, 1])
global_health_df["readmitted"] = global_health_df["readmitted"].replace(["NO", "<30", ">30"], 
                                                                     [0, 1, 2])                                                                                                                                                                                                                                                                                                                                                                                                      
global_save_dataset(global_health_df, "classification_health_encoded")