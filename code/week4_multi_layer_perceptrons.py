from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import WEEK_4_FOLDER, save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER
from week2_study import plot_evaluation_results

def ml_perceptrons(train,test, target, image_folder, filename,filename_best,filename_final):

    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    lr_type = ['constant', 'invscaling', 'adaptive']
    max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
    learning_rate = [.1, .5, .9]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(lr_type)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(lr_type)):
        d = lr_type[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in max_iter:
                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)
                mlp.fit(trnX, trnY)
                prdY = mlp.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues
        multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',xlabel='mx iter', ylabel='accuracy', percentage=True)
    save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)
    print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    if len(labels) > 2:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param= 'micro')
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(image_folder, WEEK_4_FOLDER, filename_best, show_flag=False)

    lr_type = 'adaptive'
    lr = 0.9
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in max_iter:
        mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
        mlp.fit(trnX, trnY)
        prd_tst_Y = mlp.predict(tstX)
        prd_trn_Y = mlp.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    
    plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr episodes', ylabel=str(eval_metric))
    save_image(image_folder, WEEK_4_FOLDER, filename_final, show_flag=False)

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)


ml_perceptrons(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "ml_perceptrons","ml_perceptrons_best","ml_perceptrons_final")

ml_perceptrons(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "ml_perceptrons","ml_perceptrons_best","ml_perceptrons_final")
