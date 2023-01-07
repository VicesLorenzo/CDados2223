from globals import save_image, WEEK_2_FOLDER
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pandas import unique
from matplotlib.pyplot import figure
from ds_charts import multiple_line_chart
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def nb_study(df, target_class, image_folder, filename):
    data = df.copy(deep=True)
    y = data.pop(target_class).values
    X = data.values
    labels = unique(y)
    labels.sort()
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    if len(labels) > 2:
        clf = MultinomialNB()
    else:
        clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,0], )
    plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,1], normalize=True)
    plt.tight_layout()
    save_image(image_folder, WEEK_2_FOLDER, filename, show_flag=False)

def knn_study(df, target_class, image_folder, filename):
    data = df.copy(deep=True)
    y = data.pop(target_class).values
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

def plot_confusion_matrix(cnf_matrix: np.ndarray, classes_names: np.ndarray, ax: plt.Axes = None, normalize: bool = False):
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