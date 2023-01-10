from numpy import ndarray
from pandas import unique
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ds_charts import HEIGHT, multiple_bar_chart, plot_confusion_matrix
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import WEEK_4_FOLDER, save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from matplotlib.pyplot import Axes, gca, figure, savefig, subplots, imshow, imread, axis

def nb_model(train, test, target_class, image_folder, filename):
    trnY: ndarray = train.pop(target_class).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()
    labels = labels.flatten()

    tstY: ndarray = test.pop(target_class).values
    tstX: ndarray = test.values

    if len(labels) > 2:
        clf = MultinomialNB()
    else:
        clf = GaussianNB()
    
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    if len(labels) > 2:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param= 'micro')
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    
    save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)

def plot_evaluation_results(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst, pos_value: int = 1, average_param: str = 'binary'):

    def compute_eval(real, prediction):
        evaluation = {
            'acc': accuracy_score(real, prediction),
            'recall': recall_score(real, prediction, pos_label=pos_value, average=average_param),
            'precision': precision_score(real, prediction, pos_label=pos_value, average=average_param),
            'f1': f1_score(real, prediction, pos_label=pos_value, average=average_param)
        }
        return evaluation

    eval_trn = compute_eval(trn_y, prd_trn)
    eval_tst = compute_eval(tst_y, prd_tst)
    evaluation = {}
    for key in eval_trn.keys():
        evaluation[key] = [eval_trn[key], eval_tst[key]]

    _, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets", percentage=True)

    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels= labels)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')


nb_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "nb_model")

nb_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "nb_model")