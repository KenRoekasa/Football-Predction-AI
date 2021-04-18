import sys
sys.path.append('.')

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# decision tree for feature importance on a classification problem
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define dataset
from sklearn.model_selection import train_test_split
from data_preparation.dataloader import load_training_data, normalise_input_array

selected_features = ['score', 'total shots', 'total conversion rate', 'open play shots', 'open play goals',
                     'open play conversion rate', 'set piece shots', 'set piece goals', 'set piece conversion',
                     'counter attack shots', 'counter attack goals', 'counter attack conversion', 'total passes',
                     'total average pass streak', 'crosses', 'crosses average pass streak', 'through balls',
                     'through balls average streak', 'long balls', 'long balls average streak', 'short passes',
                     'short passes average streak', 'fouls', 'red cards', 'yellow cards', 'cards per foul', 'woodwork',
                     'shots on target', 'shots off target', 'shots blocked', 'possession', 'touches', 'passes success',
                     'accurate passes', 'key passes', 'dribbles won', 'dribbles attempted', 'dribbled past',
                     'dribble success', 'aerials won', 'aerials won%', 'offensive aerials', 'defensive aerials',
                     'successful tackles', 'tackles attempted', 'was dribbled', 'tackles success %', 'clearances',
                     'interceptions', 'corners', 'corner accuracy', 'dispossessed', 'errors', 'offsides',
                     'goals conceded',
                     'win streak', 'lose streak', 'elo', 'pi rating']

x, y, features = load_training_data('../../data/whoscored/trainingdata/sum/alltrainingdata-6.csv',
                                    selected_features, 'all', True)

x = normalise_input_array(x, 'ratio')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=12)

# oversample
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# define the model
model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy')
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()
print('Random Forest')
# summarize feature importance

with open('logs/randomforest.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
pyplot.show()

# # define the model
# model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy')
# # fit the model
# model.fit(X_train, y_train)
# results = permutation_importance(model, X_test, y_test, scoring='accuracy')
# # get importance
# importance = results.importances_mean
# # summarize feature importance
#
# with open('logs/randomforestpermutation_minmax.csv', 'w+') as file:
#     for i in sorted_importance_index:
#         file.write('%s,%f' % (features[i], importance[i]))
#         file.write('\n')
#         print('%s: Score: %.5f' % (features[i], importance[i]))
#
# pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
# pyplot.show()




# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('logs/decisiontree.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
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
# with open('logs/kneighbour.csv', 'w+') as file:
#     for i in sorted_importance_index:
#         file.write('%s,%f' % (features[i], importance[i]))
#         file.write('\n')
#         print('%s: Score: %.5f' % (features[i], importance[i]))
#
# pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
# pyplot.show()

# model = LogisticRegression(random_state=0, max_iter=600)
# # fit the model
# model.fit(X_train, y_train)
# # get importance
# importance = model.coef_[0]
# sorted_importance_index = importance.argsort()
#
# with open('logs/losgistic regression.csv', 'w+') as file:
#     for i in sorted_importance_index:
#         file.write('%s,%f' % (features[i], importance[i]))
#         file.write('\n')
#         print('%s: Score: %.5f' % (features[i], importance[i]))
#
# pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
# pyplot.show()

model = XGBClassifier(use_label_encoder=False)

# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('logs/XGBClassifier.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
pyplot.show()

model = ExtraTreesClassifier(min_samples_split=2, min_samples_leaf=4, max_depth=20, criterion='entropy')
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
sorted_importance_index = importance.argsort()

with open('logs/extratreeclassifer.csv', 'w+') as file:
    for i in sorted_importance_index:
        file.write('%s,%f' % (features[i], importance[i]))
        file.write('\n')
        print('%s: Score: %.5f' % (features[i], importance[i]))

pyplot.barh(np.array(features)[sorted_importance_index], np.array(importance)[sorted_importance_index])
pyplot.show()
