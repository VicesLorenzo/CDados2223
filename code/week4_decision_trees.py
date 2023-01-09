from numpy import ndarray, argsort, arange
from pandas import unique
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import figure, subplots
from sklearn.metrics import accuracy_score
from ds_charts import horizontal_bar_chart, plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from sklearn import tree

def decision_trees_model(train, test, target_class):
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
        #save_image here
    
    labels = [str(value) for value in labels]
    tree.plot_tree(best_model, feature_name=train.columns, class_names=labels)
    #save_image here

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    #save_image here

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
    #save_image here

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
    #save_image here
    

decision_trees_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET)

decision_trees_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET)