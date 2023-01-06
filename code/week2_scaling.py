from globals import class_climate_df, class_health_df
from globals import class_climate_encoded_df, class_health_encoded_df
from globals import save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from globals import class_health_df
from globals import class_health_encoded_df
from globals import save_image, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pandas import DataFrame, concat, unique
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure
from ds_charts import get_variable_types, multiple_line_chart
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'int64':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types

def scaling(data, image_folder, filename):
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    def zscore_scaling(df_nr, df_sb, df_bool):
        transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
        tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
        norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
        return norm_data_zscore

    def minmax_scaling(df_nr, df_sb, df_bool):
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
        norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
        return norm_data_minmax

    norm_data_zscore = zscore_scaling(df_nr, df_sb, df_bool)
    norm_data_minmax = minmax_scaling(df_nr, df_sb, df_bool)

    if(image_folder == HEALTH_IMAGE_FOLDER):
        plot_nb_study(norm_data_zscore, HEALTH_IMAGE_FOLDER, "nb_scaling")
        plot_nb_study(norm_data_minmax, HEALTH_IMAGE_FOLDER, "nb_scaling")
        plot_knn_study(norm_data_zscore, HEALTH_IMAGE_FOLDER, "nb_scaling")
        plot_knn_study(norm_data_minmax, HEALTH_IMAGE_FOLDER, "nb_scaling")
    elif (image_folder == CLIMATE_IMAGE_FOLDER):
        plot_nb_study(norm_data_zscore, CLIMATE_IMAGE_FOLDER, "nb_scaling")
        plot_nb_study(norm_data_minmax, CLIMATE_IMAGE_FOLDER, "nb_scaling")
        plot_knn_study(norm_data_zscore, CLIMATE_IMAGE_FOLDER, "nb_scaling")
        plot_knn_study(norm_data_minmax, CLIMATE_IMAGE_FOLDER, "nb_scaling")

    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    data.boxplot(ax=axs[0, 0])
    axs[0, 1].set_title('Z-score normalization')
    norm_data_zscore.boxplot(ax=axs[0, 1])
    axs[0, 2].set_title('MinMax normalization')
    norm_data_minmax.boxplot(ax=axs[0, 2])
    save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)

def plot_confusion_matrix(cnf_matrix: np.ndarray, classes_names: np.ndarray, ax: plt.Axes = None,
                          normalize: bool = False):

    CMAP = plt.cm.Blues

    if ax is None:
        ax = plt.gca()
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), color='y', horizontalalignment="center")



def plot_nb_study(df, image_folder, filename):

    data = df


    if(image_folder == HEALTH_IMAGE_FOLDER):
        y = data.pop('readmitted').values
    elif (image_folder == CLIMATE_IMAGE_FOLDER):
        y = data.pop('drought').values
    X = data.values
    labels = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,0], )
    plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,1], normalize=True)
    plt.tight_layout()
    plt.show()
    save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)

def plot_knn_study(df, image_folder, filename):
    data = df

    if(image_folder == HEALTH_IMAGE_FOLDER):
        y = data.pop('readmitted').values
    elif (image_folder == CLIMATE_IMAGE_FOLDER):
        y = data.pop('drought').values
    X = data.values
    labels = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    eval_metric = accuracy_score
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        y_tst_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
        values[d] = y_tst_values

        figure()
        multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
        save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)


register_matplotlib_converters()

scaling(class_climate_encoded_df, CLIMATE_IMAGE_FOLDER, "scaling_encoded")
scaling(class_health_encoded_df, HEALTH_IMAGE_FOLDER, "scaling_encoded")