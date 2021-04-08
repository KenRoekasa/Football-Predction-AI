import csv
import sys

import numpy
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append('../..')

# random forest for feature importance on a regression problem
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from matplotlib import pyplot
# decision tree for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define dataset
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from data_preparation.dataloader import load_training_data, normalise_input_array
from model.config import COLUMNS

x, y, features = load_training_data('../../data/whoscored/trainingdata/sum/alltrainingdata-6.csv',
                                    [], 'all', True)


x = normalise_input_array(x, 'ratio')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=12)

# oversample
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

# define the model
model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy')
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()
print('Random Forest')
# summarize feature importance

with open('randomforest.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
pyplot.show()



# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('decisiontree.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
pyplot.show()


# # define the model
# model = KNeighborsClassifier()
# # fit the model
# model.fit(X_train, y_train)
# # perform permutation importance
# results = permutation_importance(model, X_test, y_test, scoring='accuracy')
# # get importance
# importance = results.importances_mean
# # summarize feature importance
#
# with open('kneighbour.csv', 'w+') as file:
#     for i in sorted_importance_index:
#         file.write('%s,%f' % (features[i], importance[i]))
#         file.write('\n')
#         print('%s: Score: %.5f' % (features[i], importance[i]))
#
# pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
# pyplot.show()


model = LogisticRegression(random_state=0, max_iter=600)
# fit the model
model.fit(X_train, y_train)
# get importance
importance =  model.coef_[0]
sorted_importance_index = importance.argsort()

with open('losgistic regression.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
pyplot.show()

model = XGBClassifier(use_label_encoder=False)

# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('XGBClassifier.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
pyplot.show()

model = ExtraTreesClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy')
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('extratreeclassifer.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(numpy.array(features)[sorted_importance_index], numpy.array(importance)[sorted_importance_index])
pyplot.show()
