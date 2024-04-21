#Import Python Packages
#from google.colab import drive
#drive.mount('/content/drive/')
from google.colab import drive
drive.mount('/gdrive')

#Import all necessary library
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import DataConversionWarning

#Read training data file
trainfile = r'/gdrive/MyDrive/ColabNotebooks/IFTrain.csv'
trainData = pd.read_csv(trainfile)
r'gdrive/MyDrive/BrainMRIimages/data'
#Read test data file
testfile = r'/gdrive/MyDrive/ColabNotebooks/IFTest.csv'
testData = pd.read_csv(testfile)

trainData.head()
#print("=======")
#testData.head()

#To get list of names of all Columns from a dataframe
TrainCols = list(trainData.columns.values)
TestCols = list(testData.columns.values)
print(TrainCols)
print(TestCols)

label_encoder = LabelEncoder()
for i in range(len(TrainCols)):
  trainData[TrainCols[i]] = label_encoder.fit_transform(trainData[TrainCols[i]])
for i in range(len(TestCols)):
  testData[TestCols[i]] = label_encoder.fit_transform(testData[TestCols[i]])

print(testData.head())

# Seperate Target column from Train Data
Xtrain = trainData[TrainCols[0:len(TrainCols)-1]].copy()
Ytrain = trainData[['FRAUDFOUND']].copy()
print(Xtrain.shape)
print(Ytrain.shape)
Xtest = testData[TestCols[0:len(TestCols)-1]].copy()
Ytest = testData[['FRAUDFOUND']].copy()
print(Xtest.shape)
print(Ytrain.value_counts())
print(Ytest.value_counts())

# trainData['TARGET'].hist()
# plt.xlabel("Target Score")
# plt.ylabel("Frequency")
# plt.show()
class_distribution = Ytrain.value_counts()
print(class_distribution)
import matplotlib.pyplot as plt
class_distribution.plot(kind='bar')
plt.title('Target Score Histogram')
plt.xlabel('Target scores')
plt.ylabel('Frequency')
plt.show()

clf = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(5,50,10)),
    'min_samples_split': list(range(2,10,2)),
    'min_samples_leaf': list(range(1,6,2))
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
grid_search.fit(Xtrain, Ytrain)
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy Score: ", grid_search.best_score_)
grid_param = grid_search.best_params_

best_clf = DecisionTreeClassifier(**grid_param)
best_clf.fit(Xtrain, Ytrain)
Ypred = best_clf.predict(Xtest)
print(confusion_matrix(Ytest,Ypred))
print(classification_report(Ytest,Ypred))
print(accuracy_score(Ytest, Ypred))
print(roc_auc_score(Ytest,Ypred))
clf_cv_score = cross_val_score(best_clf, Xtest, Ytest, cv=10, scoring="roc_auc")
print(clf_cv_score.mean())

rand_clf = DecisionTreeClassifier()
rand_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(5,50)),
    'min_samples_split': list(range(2,10)),
    'min_samples_leaf': list(range(1,6))
}
rand_search = RandomizedSearchCV(rand_clf, rand_grid, n_iter=15, cv=5, n_jobs=2)
rand_search.fit(Xtrain, Ytrain)
print("Best Parameters: ", rand_search.best_params_)
print("Best Accuracy Score: ", rand_search.best_score_)
rand_param = rand_search.best_params_

best_clf_rand = DecisionTreeClassifier(**rand_param)
best_clf_rand.fit(Xtrain, Ytrain)
Ypred2 = best_clf_rand.predict(Xtest)
print(confusion_matrix(Ytest,Ypred2))
print(classification_report(Ytest,Ypred2))
print(accuracy_score(Ytest, Ypred2))
print(roc_auc_score(Ytest,Ypred2))
clf_cv_score2 = cross_val_score(best_clf_rand, Xtest, Ytest, cv=10, scoring="roc_auc")
print(clf_cv_score.mean())

param_grid1 = {'max_depth' : list(range(3,11,2)),
              'min_samples_split': list(range(2,11,1)),
              'min_samples_leaf' : list(range(2,10,2))}
clf1 = RandomForestClassifier(random_state=0)
grid_search1 = GridSearchCV(estimator=clf1, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
grid_search1.fit(Xtrain, Ytrain)
print("Best Parameters: ", grid_search1.best_params_)
print("Best Accuracy Score: ", grid_search1.best_score_)
grid_param1 = grid_search1.best_params_

import warnings

# Suppress DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
best_clf1 = RandomForestClassifier(**grid_param1)
best_clf1.fit(Xtrain, Ytrain)
Ypred1 = best_clf1.predict(Xtest)
print(confusion_matrix(Ytest,Ypred1))
print(classification_report(Ytest,Ypred1))
print(accuracy_score(Ytest, Ypred1))
print(roc_auc_score(Ytest,Ypred))
clf_cv_score1 = cross_val_score(best_clf1, Xtest, Ytest, cv=10, scoring="roc_auc")
print(clf_cv_score1.mean())

rand_grid1 = {'max_depth' : list(range(3,11,2)),
              'min_samples_split': list(range(2,15,1)),
              'max_leaf_nodes' : list(range(2,15,2))}
clf1 = RandomForestClassifier()
grid_search1 = GridSearchCV(estimator=clf1, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
grid_search1.fit(Xtrain, Ytrain)
print("Best Parameters: ", grid_search1.best_params_)
print("Best Accuracy Score: ", grid_search1.best_score_)
grid_param1 = grid_search1.best_params_

import warnings

# Suppress DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
best_clf1 = RandomForestClassifier(**grid_param1)
best_clf1.fit(Xtrain, Ytrain)
Ypred1 = best_clf1.predict(Xtest)
print(confusion_matrix(Ytest,Ypred1))
print(classification_report(Ytest,Ypred1))
print(accuracy_score(Ytest, Ypred1))
clf_cv_score1 = cross_val_score(best_clf1, Xtest, Ytest, cv=10, scoring="roc_auc")
print(clf_cv_score1.mean())
