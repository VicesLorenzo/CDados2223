from numpy import ndarray, argsort, arange
from pandas import unique
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.pyplot import figure, subplots
from sklearn.metrics import accuracy_score
from ds_charts import horizontal_bar_chart, plot_evaluation_results, multiple_line_chart
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER
from globals import save_image, WEEK_4_FOLDER
from week2_study import plot_evaluation_results

def decision_trees_model(train, test, target_class, image_folder, filename, filename2, filename3, filename4, filename5, filename6):
    trnY: ndarray = train.pop(target_class).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = test.pop(target_class).values
    tstX: ndarray = test.values

    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_model = None

    figure()
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                           xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
        if(k == 0):
            save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)
        else:
            save_image(image_folder, WEEK_4_FOLDER, filename2, show_flag=False)

    labels = [str(value) for value in labels]
    plot_tree(best_model, feature_names= train.columns, class_names=labels)
    save_image(image_folder, WEEK_4_FOLDER, filename3, show_flag=False)

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    if len(labels) > 2:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param= 'micro')
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(image_folder, WEEK_4_FOLDER, filename4, show_flag=False)

    variables = train.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]

    figure()
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
    save_image(image_folder, WEEK_4_FOLDER, filename5, show_flag=False)

    imp = 0.0001
    f = 'entropy'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for d in max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(trnX, trnY)
        prdY = tree.predict(tstX)
        prd_tst_Y = tree.predict(tstX)
        prd_trn_Y = tree.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))
    save_image(image_folder, WEEK_4_FOLDER, filename6, show_flag=False)
    
def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)

decision_trees_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, 
"dtrees_entropy_multiline", "dtrees_gini_multiline", "dtrees_tree", "dtrees_plot_eval", "dtrees_horizontal_bar", "dtrees_plot_overfit")

decision_trees_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER,
"dtrees_entropy_multiline", "dtrees_gini_multiline", "dtrees_tree", "dtrees_plot_eval", "dtrees_horizontal_bar", "dtrees_plot_overfit")