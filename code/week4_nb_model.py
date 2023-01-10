from numpy import ndarray
from pandas import unique
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ds_charts import plot_evaluation_results
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET
from globals import WEEK_4_FOLDER, save_image, CLIMATE_IMAGE_FOLDER, HEALTH_IMAGE_FOLDER

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
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(image_folder, WEEK_4_FOLDER, filename, show_flag=False)


#nb_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET, CLIMATE_IMAGE_FOLDER, "nb_model")

nb_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET, HEALTH_IMAGE_FOLDER, "nb_model")