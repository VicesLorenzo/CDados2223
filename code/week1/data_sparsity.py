from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show, figure, savefig, show, title
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap
import matplotlib
from operator import itemgetter

matplotlib.use('Agg')


register_matplotlib_converters()
filename_health = './datasets/classification_health.csv'
filename_climate = './datasets/classification_climate.csv'


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


def path_to_images(folder, filename, ext="png"):
    return f"./images/week1/{folder}/{filename}.{ext}"


def run(index):
    data, images_path = itemgetter("data", "images_path")(DB[index])

    def numeric():
        numeric_vars = get_variable_types(data)['Numeric']
        if [] == numeric_vars:
            return
            # raise ValueError('There are no numeric variables.')

        rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
        fig, axs = subplots(rows, cols, figsize=(
            cols*HEIGHT, rows*HEIGHT), squeeze=False)
        for i in range(len(numeric_vars)):
            var1 = numeric_vars[i]
            for j in range(i+1, len(numeric_vars)):
                var2 = numeric_vars[j]
                axs[i, j-1].set_title("%s x %s" % (var1, var2))
                axs[i, j-1].set_xlabel(var1)
                axs[i, j-1].set_ylabel(var2)
                axs[i, j-1].scatter(data[var1], data[var2])
        savefig(path_to_images(
            f'{images_path}/sparsity', "sparsity_study_numeric"))
        # show()

    def symbolic():
        symbolic_vars = get_variable_types(data)['Symbolic']

        print(symbolic_vars)
        if [] == symbolic_vars:
            return
            # raise ValueError('There are no symbolic variables.')

        rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
        fig, axs = subplots(rows, cols, figsize=(
            cols*HEIGHT, rows*HEIGHT), squeeze=False)
        for i in range(len(symbolic_vars)):
            var1 = symbolic_vars[i]
            for j in range(i+1, len(symbolic_vars)):
                var2 = symbolic_vars[j]
                axs[i, j-1].set_title("%s x %s" % (var1, var2))
                axs[i, j-1].set_xlabel(var1)
                axs[i, j-1].set_ylabel(var2)
                axs[i, j-1].scatter(data[var1], data[var2])
        savefig(path_to_images(
            f'{images_path}/sparsity', "sparsity_study_symbolic"))
        # show()

    def correlation():
        corr_mtx = abs(data.corr())
        fig = figure(figsize=[12, 12])

        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns,
                yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        title('Correlation analysis')
        savefig(path_to_images(f'{images_path}/correlation_analysis', "image"))
        # show()

    # numeric()
    symbolic()
    # correlation()


def main():

    for index in range(len(DB)):
        run(index)
    # run(1)


if __name__ == "__main__":
    main()
