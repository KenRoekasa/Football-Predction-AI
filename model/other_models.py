import sys

from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.utils import class_weight

sys.path.append('..')
import random
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import preprocessing, neighbors, metrics
from sklearn.cluster import k_means, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from data_preparation.dataloader import load_training_data, normalise_input_array
from graphing.confusion_matrix import plot_confusion_matrix as my_confusion


def evaluate_model(clf, X_train, y_train, X_test, y_test, title):
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    model_string = str(clf)[:str(clf).index('(')]
    print('{}: train {:.2%} , test {:.2%}'.format(model_string, train_accuracy, test_accuracy))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=['Win', 'Draw', 'Lose'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title(title)


    plt.savefig('../plots/baseline/'+title+'.png')


# print(clf.predict(X_test))

# max_depth = [5, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
#
# random_grid = {
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf, }
# # print(random_grid)
#
# model_tuned = RandomizedSearchCV(estimator=clf, param_distributions=random_grid)
#
# search = model_tuned.fit(X_train, y_train)
# print(search.best_params_)


if __name__ == '__main__':
    features = ['score', 'total shots', 'total conversion rate', 'open play shots', 'open play goals',
                'open play conversion rate', 'set piece shots', 'set piece goals', 'set piece conversion',
                'counter attack shots', 'counter attack goals', 'counter attack conversion', 'total passes',
                'total average pass streak', 'crosses', 'crosses average pass streak', 'through balls',
                'through balls average streak', 'long balls', 'long balls average streak', 'short passes',
                'short passes average streak', 'fouls', 'red cards', 'yellow cards', 'cards per foul', 'woodwork',
                'shots on target', 'shots off target', 'shots blocked', 'possession', 'touches', 'passes success',
                'accurate passes', 'key passes', 'dribbles won', 'dribbles attempted', 'dribbled past',
                'dribble success', 'aerials won', 'aerials won%', 'offensive aerials', 'defensive aerials',
                'successful tackles', 'tackles attempted', 'was dribbled', 'tackles success %', 'clearances',
                'interceptions', 'corners', 'corner accuracy', 'dispossessed', 'errors', 'offsides', 'goals conceded',
                'win streak', 'lose streak', 'elo', 'pi rating']

    x, y = load_training_data('../data/whoscored/trainingdata/mean/alltrainingdata-10.csv',
                              features, 'all')

    x = normalise_input_array(x, 'ratio')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=12)

    # oversample = SMOTE()
    # X_train, y_train = oversample.fit_resample(X_train, y_train)

    print(y_train.tolist().count(0))
    print(y_train.tolist().count(1))
    print(y_train.tolist().count(2))

    #
    # correct_counter = 0
    # test_pred = []
    # for y in y_test:
    #     predict = random.randint(0, 2)
    #     test_pred.append(predict)
    #     if predict == int(y):
    #         correct_counter += 1
    #
    # accuracy = correct_counter / len(y_test)
    # print('Random guessing: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))
    #
    # cm = metrics.confusion_matrix(y_test, test_pred)
    # # Log the confusion matrix as an image summary.
    # figure = my_confusion(cm, class_names=['Win', 'Draw', 'Lose'])
    # figure.show()

    evaluate_model(DummyClassifier(strategy='uniform'), X_train, y_train, X_test, y_test,'Random guessing')
    evaluate_model(DummyClassifier(strategy='stratified'), X_train, y_train, X_test, y_test,'Stratified Random Guessing')
    evaluate_model(DummyClassifier(strategy='constant', constant=0), X_train, y_train, X_test, y_test,'Wins only Guessing')
    evaluate_model(DummyClassifier(strategy='constant', constant=1), X_train, y_train, X_test, y_test,'Draws only Guessing')
    evaluate_model(DummyClassifier(strategy='constant', constant=2), X_train, y_train, X_test, y_test,'Losses only Guesses')



    evaluate_model(neighbors.KNeighborsClassifier(), X_train, y_train, X_test, y_test,'k-nearest neighbours')

    evaluate_model(RandomForestClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy',
                                          ),
                   X_train, y_train, X_test, y_test,'Random Forest')

    # clf = RandomForestClassifier(class_weight=class_weight)
    # max_depth = [5, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    #
    # random_grid = {
    #     'max_depth': max_depth,
    #     'min_samples_split': min_samples_split,
    #     'min_samples_leaf': min_samples_leaf, }
    # # print(random_grid)
    #
    # model_tuned = RandomizedSearchCV(estimator=clf, param_distributions=random_grid)
    #
    # search = model_tuned.fit(X_train, y_train)
    # print(search.best_params_)

    evaluate_model(LogisticRegression(random_state=0, max_iter=500), X_train, y_train, X_test, y_test,'Logistic Regression')

    evaluate_model(XGBClassifier(use_label_encoder=False), X_train, y_train, X_test, y_test,'XGB Classifier')

    evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test,'Decision Tree')
