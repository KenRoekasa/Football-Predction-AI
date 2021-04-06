import sys

from imblearn.under_sampling import RandomUnderSampler
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


def evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    model_string = str(clf)[:str(clf).index('(')]
    print('{}: train {:.2%} , test {:.2%}'.format(model_string, train_accuracy, test_accuracy))

    # Plot non-normalized confusion matrix
    titles_options = [("%s Confusion matrix, without normalization" % model_string, None),
                      ("%s Normalized confusion matrix" % model_string, 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=['Win', 'Draw', 'Lose'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


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
    features = ['score'
                ]

    x, y = load_training_data('../data/whoscored/trainingdata/mean/alltrainingdata-10.csv',
                              [], 'all')

    x = normalise_input_array(x, 'ratio')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=12)

    # oversample = SMOTE()
    # X_train, y_train = oversample.fit_resample(X_train, y_train)

    print(y_train.tolist().count(0))
    print(y_train.tolist().count(1))
    print(y_train.tolist().count(2))

    correct_counter = 0
    for y in y_test:
        predict = random.randint(0, 3)
        if predict == int(y):
            correct_counter += 1

    accuracy = correct_counter / len(y_test)
    print('Random guessing: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))

    correct_counter = 0
    for y in y_test:
        predict = random.randint(0, 1)
        if predict == 1:
            predict = 2
        if predict == int(y):
            correct_counter += 1

    accuracy = correct_counter / len(y_test)
    print('Random guessing without draws: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))

    correct_counter = 0
    for y in y_test:
        if int(y) == 0:
            correct_counter += 1

    accuracy = correct_counter / len(y_test)
    print('Wins only guessing: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))

    correct_counter = 0
    for y in y_test:
        if int(y) == 1:
            correct_counter += 1

    accuracy = correct_counter / len(y_test)
    print('Draw only guessing: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))

    correct_counter = 0
    for y in y_test:
        if int(y) == 2:
            correct_counter += 1

    accuracy = correct_counter / len(y_test)
    print('Lose only guessing: {0:.2%} with {1}/{2}'.format(accuracy, correct_counter, len(y_test)))

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    results = kmeans.predict(X_test)
    counter = 0
    for idx, y in enumerate(results):
        if y == y_test[idx]:
            counter += 1
    accuracy = counter / len(y_test)
    print('Kmean clustering: {0:.2%} with {1}/{2}'.format(accuracy, counter, len(y_test)))


    evaluate_model(neighbors.KNeighborsClassifier(), X_train, y_train, X_test, y_test)


    evaluate_model(RandomForestClassifier(min_samples_split= 2, min_samples_leaf= 4, max_depth=20, criterion='entropy',
                                          ),
                   X_train, y_train, X_test, y_test)



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



    evaluate_model(LogisticRegression(random_state=0, max_iter=500), X_train, y_train, X_test, y_test)


    evaluate_model(XGBClassifier(use_label_encoder=False), X_train, y_train, X_test, y_test)

    evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
