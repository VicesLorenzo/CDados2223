from numpy import ndarray
from pandas import unique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ds_charts import multiple_line_chart
from matplotlib.pyplot import figure
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import WEEK_4_FOLDER, save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER
from week2_study import plot_evaluation_results


def knn_model(train, test, target_class, image_folder, filename, filename2, filename3):
    trnY: ndarray = train.pop(target_class).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = test.pop(target_class).values
    tstX: ndarray = test.values

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
    save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)


    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    if len(labels) > 2:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param= 'micro')
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(image_folder, WEEK_4_FOLDER, filename2, show_flag=False)


    def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)

    d = 'euclidean'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        prd_trn_Y = knn.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))
    save_image(image_folder, WEEK_4_FOLDER, filename3, show_flag=False)


knn_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "knn_model_multiline",
 "knn_model_plot_eval", "knn_model_plot_overfit")

knn_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "knn_model_multiline", 
 "knn_model_plot_eval", "knn_model_plot_overfit")