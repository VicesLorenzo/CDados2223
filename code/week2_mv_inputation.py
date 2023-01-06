from globals import class_health_df
from globals import class_health_encoded_df
from globals import save_image, HEALTH_IMAGE_FOLDER, WEEK_2_FOLDER
import pandas as pd
import matplotlib.pyplot as plt
import ds_charts as ds
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pandas import DataFrame, concat, unique
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure
from ds_charts import bar_chart, get_variable_types, multiple_line_chart
from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def missing_values(df):
    mv = {}
    for var in df:
        nr = df[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    if not mv:
        return
    figure()
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
    return mv

def drop_columns(mv, data):
    # defines the number of records to discard entire columns
    threshold = data.shape[0] * 0.37
    missings = [c for c in mv.keys() if mv[c]>threshold]
    data = data.drop(columns=missings, inplace=False)
    return data

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

def value_inputation(data):
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    df = concat([tmp_nr, tmp_bool], axis=1)
    df.index = data.index
    df.describe(include='all')

    return df



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


    y = data.pop('readmitted').values
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

    y = data.pop('readmitted').values
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


plot_nb_study(value_inputation(drop_columns(missing_values(class_health_encoded_df), class_health_encoded_df)), HEALTH_IMAGE_FOLDER, "missing_values_encoded_nb_inputation")

plot_knn_study(value_inputation(drop_columns(missing_values(class_health_encoded_df), class_health_encoded_df)), HEALTH_IMAGE_FOLDER, "missing_values_encoded__knn_inputation")