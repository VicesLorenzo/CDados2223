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

FILENAME_HEALTH = HEALTH_DATASET_FOLDER + "classification_health.csv"
FILENAME_HEALTH_ENCODED = HEALTH_DATASET_FOLDER + "classification_health_encoded.csv"

FILENAME_CLIMATE = CLIMATE_DATASET_FOLDER + "classification_climate.csv"
FILENAME_CLIMATE_ENCODED = CLIMATE_DATASET_FOLDER + "classification_climate_encoded.csv"

def save_dataset(df, dataset_folder, filename):
    df.to_csv(dataset_folder + filename + ".csv", index=False)
    print(f"Saved dataset {dataset_folder + filename}.csv")

def save_image(image_folder, week_folder, filename, show_flag=False):
    savefig(image_folder + week_folder + filename + ".png")
    if show_flag:
        show()
    print(f"Saved image {image_folder + week_folder + filename}.png")

class_health_df = read_csv(FILENAME_HEALTH)
class_health_encoded_df = read_csv(FILENAME_HEALTH_ENCODED)

class_climate_df = read_csv(FILENAME_CLIMATE, parse_dates=["date"], infer_datetime_format=True, dayfirst=True)
class_climate_encoded_df = read_csv(FILENAME_CLIMATE_ENCODED, parse_dates=["date"], infer_datetime_format=True, dayfirst=True)
