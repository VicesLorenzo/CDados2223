from pandas import read_csv
from matplotlib.pyplot import savefig, show

CLIMATE_DATASET_FOLDER = "../datasets/climate/"
HEALTH_DATASET_FOLDER = "../datasets/health/"

CLIMATE_IMAGE_FOLDER = "../images/climate/"
HEALTH_IMAGE_FOLDER = "../images/health/"

WEEK_1_FOLDER = "week1/"
WEEK_2_FOLDER = "week2/"
WEEK_3_FOLDER = "week3/"
WEEK_4_FOLDER = "week4/"
WEEK_5_FOLDER = "week5/"
WEEK_6_FOLDER = "week6/"
WEEK_7_FOLDER = "week7/"

CLASSIFICATION_HEALTH_TARGET = "readmitted"
CLASSIFICATION_CLIMATE_TARGET = "class"

CLASSIFICATION_HEALTH_FILENAME = "classification_health"
CLASSIFICATION_HEALTH_ENCODED_FILENAME = "classification_health_encoded"
CLASSIFICATION_HEALTH_PREPARED_FILENAME = "classification_health_prepared"
CLASSIFICATION_HEALTH_PREPARED_TRAIN_FILENAME = "classification_health_prepared_train"
CLASSIFICATION_HEALTH_PREPARED_TEST_FILENAME = "classification_health_prepared_test"
FORECASTING_HEALTH_FILENAME = "forecasting_health"

CLASSIFICATION_CLIMATE_FILENAME = "classification_climate"
CLASSIFICATION_CLIMATE_ENCODED_FILENAME = "classification_climate_encoded"
CLASSIFICATION_CLIMATE_PREPARED_FILENAME = "classification_climate_prepared"
CLASSIFICATION_CLIMATE_PREPARED_TRAIN_FILENAME = "classification_climate_prepared_train"
CLASSIFICATION_CLIMATE_PREPARED_TEST_FILENAME = "classification_climate_prepared_test"
FORECASTING_CLIMATE_FILENAME = "forecasting_climate"

FILENAME_HEALTH_PATH = HEALTH_DATASET_FOLDER + CLASSIFICATION_HEALTH_FILENAME + ".csv"
FILENAME_HEALTH_ENCODED_PATH = HEALTH_DATASET_FOLDER + CLASSIFICATION_HEALTH_ENCODED_FILENAME + ".csv"
FILENAME_HEALTH_PREPARED_PATH = HEALTH_DATASET_FOLDER + CLASSIFICATION_HEALTH_PREPARED_FILENAME + ".csv"
FILENAME_HEALTH_PREPARED_TRAIN_PATH = HEALTH_DATASET_FOLDER + CLASSIFICATION_HEALTH_PREPARED_TRAIN_FILENAME + ".csv"
FILENAME_HEALTH_PREPARED_TEST_PATH = HEALTH_DATASET_FOLDER + CLASSIFICATION_HEALTH_PREPARED_TEST_FILENAME + ".csv"
FILENAME_HEALTH_FORECASTING_PATH = HEALTH_DATASET_FOLDER + FORECASTING_HEALTH_FILENAME + ".csv"

FILENAME_CLIMATE_PATH = CLIMATE_DATASET_FOLDER + CLASSIFICATION_CLIMATE_FILENAME + ".csv"
FILENAME_CLIMATE_ENCODED_PATH = CLIMATE_DATASET_FOLDER + CLASSIFICATION_CLIMATE_ENCODED_FILENAME + ".csv"
FILENAME_CLIMATE_PREPARED_PATH = CLIMATE_DATASET_FOLDER + CLASSIFICATION_CLIMATE_PREPARED_FILENAME + ".csv"
FILENAME_CLIMATE_PREPARED_TRAIN_PATH = CLIMATE_DATASET_FOLDER + CLASSIFICATION_CLIMATE_PREPARED_TRAIN_FILENAME + ".csv"
FILENAME_CLIMATE_PREPARED_TEST_PATH = CLIMATE_DATASET_FOLDER + CLASSIFICATION_CLIMATE_PREPARED_TEST_FILENAME + ".csv"
FILENAME_CLIMATE_FORECASTING_PATH = CLIMATE_DATASET_FOLDER + FORECASTING_CLIMATE_FILENAME + ".csv"


def save_dataset(df, dataset_folder, filename):
    df.to_csv(dataset_folder + filename + ".csv", index=False)
    print(f"Saved dataset {dataset_folder + filename}.csv")

def save_image(image_folder, week_folder, filename, show_flag=False):
    savefig(image_folder + week_folder + filename + ".png")
    if show_flag:
        show()
    print(f"Saved image {image_folder + week_folder + filename}.png")

class_health_df = read_csv(FILENAME_HEALTH_PATH)
class_health_encoded_df = read_csv(FILENAME_HEALTH_ENCODED_PATH)
class_health_prepared_df = read_csv(FILENAME_HEALTH_PREPARED_PATH)
class_health_prepared_train_df = read_csv(FILENAME_HEALTH_PREPARED_TRAIN_PATH)
class_health_prepared_test_df = read_csv(FILENAME_HEALTH_PREPARED_TEST_PATH)
forecast_health_df = read_csv(FILENAME_HEALTH_FORECASTING_PATH, index_col='Date', sep=',', decimal='.', parse_dates=True, dayfirst=True)

class_climate_df = read_csv(FILENAME_CLIMATE_PATH, parse_dates=["date"], infer_datetime_format=True, dayfirst=True)
class_climate_encoded_df = read_csv(FILENAME_CLIMATE_ENCODED_PATH)
class_climate_prepared_df = read_csv(FILENAME_CLIMATE_PREPARED_PATH)
class_climate_prepared_train_df = read_csv(FILENAME_CLIMATE_PREPARED_TRAIN_PATH)
class_climate_prepared_test_df = read_csv(FILENAME_CLIMATE_PREPARED_TEST_PATH)
forecast_climate_df = read_csv(FILENAME_CLIMATE_FORECASTING_PATH, index_col='date', sep=',', decimal='.', parse_dates=True, dayfirst=True)