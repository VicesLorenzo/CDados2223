from pandas import read_csv

filename_health = '../../datasets/classification_health.csv'
filename_climate = '../../datasets/classification_climate.csv'
filename_h = "classification_health"
filename_c = "classification_climate"


def gbVar():
    data_health = read_csv(filename_health, index_col='encounter_id',
                        parse_dates=True, infer_datetime_format=True)
    data_climate = read_csv(filename_climate, index_col='date',
                            parse_dates=True, infer_datetime_format=True)

    health_symb1 = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

    conv = {x: 0 if x == "down" else 1 if x ==
        "up" else 2 if x == "steady" else 3 for x in health_symb1}


    DB = [{"data": data_health, "images_path": "health", 'symbolic_vars': ["readmitted", 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'A1Cresult'], 'symbolic_vars_converter': conv},
      {"data": data_climate, "images_path": "climate", 'symbolic_vars': []}]


    return data_health,data_climate