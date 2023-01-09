from numpy import ndarray
from pandas import unique
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ds_charts import plot_evaluation_results
from globals import class_climate_prepared_test_df, class_climate_prepared_train_df, class_health_prepared_test_df, class_health_prepared_train_df
from globals import CLASSIFICATION_CLIMATE_TARGET, CLASSIFICATION_HEALTH_TARGET

def nb_model(train, test, target_class):
    trnY: ndarray = train.pop(target_class).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

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


nb_model(class_climate_prepared_train_df, class_climate_prepared_test_df, CLASSIFICATION_CLIMATE_TARGET)

nb_model(class_health_prepared_train_df, class_health_prepared_test_df, CLASSIFICATION_HEALTH_TARGET)