from numpy import ndarray,std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import WEEK_4_FOLDER, save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER
from week2_study import plot_evaluation_results


def random_forest(train,test, target, image_folder, filename,filename_best,filename_ranking,filename_final):

    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10, 25]
    max_features = [.3, .5, .7, 1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',xlabel='nr estimators', ylabel='accuracy', percentage=True)
    save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    if len(labels) > 2:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param= 'micro')
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(image_folder, WEEK_4_FOLDER, filename_best, show_flag=False)

    variables = train.columns
    importances = best_model.feature_importances_
    stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = argsort(importances)[::-1]
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
    save_image(image_folder, WEEK_4_FOLDER, filename_ranking, show_flag=False)

    f = 0.7
    max_depth = 10
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
        rf.fit(trnX, trnY)
        prd_tst_Y = rf.predict(tstX)
        prd_trn_Y = rf.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))
    save_image(image_folder, WEEK_4_FOLDER, filename_final, show_flag=False)

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)






random_forest(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "random_forest","random_forest_best","random_forest_ranking","random_forest_final")
random_forest(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "random_forest","random_forest_best","random_forest_ranking","random_forest_final")
