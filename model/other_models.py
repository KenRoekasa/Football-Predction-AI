import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing, neighbors, metrics
from sklearn.cluster import k_means, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from data_preparation.dataloader import load_training_data, normalise_input_array

features = ['score',
                'total shots',
                'shots on target', 'win streak', 'lose streak', 'pi rating', 'elo']


x, y = load_training_data('../data/whoscored/trainingdata/sum/alltrainingdata.csv',
                          [])

x = normalise_input_array(x, 'max')


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True)

# oversample
# oversample = RandomOverSampler(sampling_strategy='minority')
# X_train, y_train = oversample.fit_resample(X_train, y_train)

print('Kmean clustering')
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

results = kmeans.predict(X_test)

counter = 0

for idx, y in enumerate(results):
    if y == y_test[idx]:
        counter += 1

accuracy = counter / len(y_test)

print('{0:.2%} with {1}/{2}'.format(accuracy, counter, len(y_test)))

print('K nearest neighbours')
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
print(accuracy)

accuracy = clf.score(X_test, y_test)
print(accuracy)

print('Random Forest')
clf = RandomForestClassifier(min_samples_split= 2, min_samples_leaf= 2, max_depth=30 ,criterion='entropy')

clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
print(accuracy)

accuracy = clf.score(X_test, y_test)
print(accuracy)

predictions = clf.predict(X_test)

print(predictions.tolist().count(0),predictions.tolist().count(1),predictions.tolist().count(2))



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


print('Logistic Regression')
model_LR = LogisticRegression(random_state=0,max_iter=500)
model_LR = model_LR.fit(X_train, y_train)

accuracy = model_LR.score(X_train, y_train)
print(accuracy)

accuracy = model_LR.score(X_test, y_test)
print(accuracy)

print('XGB Classifier')
model_XGB = XGBClassifier(use_label_encoder=False)
model_XGB.fit(X_train, y_train)

predictions = model_XGB.predict(X_test)

print(predictions.tolist().count(0),predictions.tolist().count(1),predictions.tolist().count(2))


accuracy = model_XGB.score(X_train, y_train)
print(accuracy)

accuracy = model_XGB.score(X_test, y_test)
print(accuracy)


print('Decision Tree')
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
print(accuracy)

accuracy = clf.score(X_test, y_test)
print(accuracy)
preds_DT = clf.predict(X_test)

print('Logistic Regression Confusion Matrix \n', metrics.confusion_matrix(y_test, preds_DT))