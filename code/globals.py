from pandas import DataFrame
from pandas import read_csv

DATASET_FOLDER = "../datasets/"

FILENAME_HEALTH = DATASET_FOLDER + "classification_health.csv"
FILENAME_HEALTH_ENCODED = DATASET_FOLDER + "classification_health_encoded.csv"

FILENAME_CLIMATE = DATASET_FOLDER + "classification_climate.csv"
FILENAME_CLIMATE_ENCODED = DATASET_FOLDER + "classification_climate_encoded.csv"

global_health_df = read_csv(FILENAME_HEALTH)
global_health_encoded_df = read_csv(FILENAME_HEALTH_ENCODED)

global_climate_df = read_csv(FILENAME_CLIMATE, parse_dates=["date"], infer_datetime_format=True, dayfirst=True)
global_climate_encoded_df = read_csv(FILENAME_CLIMATE_ENCODED, parse_dates=["date"], infer_datetime_format=True, dayfirst=True)

def global_save_dataset(df : DataFrame, filename):
    df.to_csv(DATASET_FOLDER + filename + ".csv", index=False)
