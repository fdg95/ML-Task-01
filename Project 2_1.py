# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:13:48 2020

This task is primarily concerned with multi-class classification where you have
 3 classes. However, we have changed the original image features in several ways. 
 You will need to deal with class imbalance; in the training set, 
 there are 600 examples from class 0 and 2 but 3600 examples from class 1. 
 Test set has the same class imbalance as the training set.

@author: Fabian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

#Import the training data
data_train = pd.read_csv('X_train.csv')
label_train = pd.read_csv('y_train.csv')
data_test = pd.read_csv('X_test.csv')
sample_sub = pd.read_csv('sample.csv')

#Drop the id-column
data_train = data_train.drop(['id'], axis = 1)
data_test = data_test.drop(['id'], axis = 1)
label_train = label_train.drop(['id'], axis = 1)

#Only acitvate early split for oversampling, so that oversampled labels don't fall into test set
#data_train, test_x, label_train, test_y = train_test_split(data_train, label_train, stratify = label_train, random_state = 42)

#Create dataframe with complete training data
data = data_train.copy()
data['y'] = label_train['y']

#%% Oversampling
max_size = data['y'].value_counts().max()
print(max_size)

lst = [data]
for class_index, group in data.groupby('y'):
    lst.append(group.sample(max_size - len(group), replace = True))
data = pd.concat(lst)

label_train = data['y']
data_train = data.drop(['y'], axis = 1)

#%%
#Get some information on our data
print(data_train.info())
print(data_train.isnull().sum())
print(data_train.describe())

#%% Feature Selection (1)
#Calculate correlation of each column with the label
correlation = data[data.columns[0:]].corr()['y'][:]
correlation.drop('y', inplace = True)

#%%Feature Selection (2)
#Create a Mask with all features with an absolute correlation exceeding certain threshold
corr = pd.DataFrame()
corr['correlation'] = correlation.abs()
corr.loc[corr['correlation'] > 0.075, 'mask'] = True
corr['mask'].fillna(False, inplace = True)

#Filter out the features with too little correlation
data_train = data_train.loc[:,corr['mask'] == 1]
#test_x = test_x.loc[:, corr['mask'] == 1]
data_test = data_test.loc[:, corr['mask'] == 1]

#%% Feature Selection (3)

#Feature selection with SelectKBest --> Choose approx. same number of features as before
fs = SelectKBest(score_func=f_regression, k = 400)
fs.fit(data_train, label_train)
data_train = fs.transform(data_train)
data_test = fs.transform(data_test)

#%%
#Splitting the training data into an actual training and a test set
train_x, test_x, train_y, test_y = train_test_split(data_train, label_train, random_state = 42)

#%%
#GridSearch for best ExtraTree Parameters
params = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10]}
xtree_gs = GridSearchCV(estimator = ExtraTreeClassifier(), scoring = 'balanced_accuracy' , verbose = 1, param_grid=params, cv = 10)
xtree_gs.fit(train_x, train_y)

xtree_best_results = xtree_gs.cv_results_
xtree_best_params = xtree_gs.best_params_
xtree_best_estimator = xtree_gs.best_estimator_
xtree_best_score = xtree_gs.best_score_

#%%
#Training an ExtraTree classifier
xtree = ExtraTreeClassifier(criterion = 'gini', max_depth = 7, class_weight='balanced')
xtree.fit(train_x, train_y)
y_pred_train = xtree.predict(train_x)
y_pred_test = xtree.predict(test_x)

#Computing training and testing scores
train_score = balanced_accuracy_score(train_y, y_pred_train)
test_score = balanced_accuracy_score(test_y, y_pred_test)
print('Training score:', train_score)
print('Testing score:', test_score)

#%%
params = {'kernel': ['rbf', 'poly'], 'C': [0.1, 1, 10], 'decision_function_shape': ['ovo', 'ovr']}
svc_gs = GridSearchCV(estimator = SVC(class_weight = 'balanced'), scoring = 'balanced_accuracy' , verbose = 1, param_grid=params, cv = 10)
svc_gs.fit(train_x, train_y)

svc_best_results = svc_gs.cv_results_
svc_best_params = svc_gs.best_params_
svc_best_estimator = svc_gs.best_estimator_
svc_best_score = svc_gs.best_score_

#%%

svm_classifier = SVC()

svc = SVC(kernel = 'rbf', C = 1, decision_function_shape='ovo', class_weight='balanced')
svc.fit(train_x, train_y)
y_pred_train = svc.predict(train_x)
y_pred_test = svc.predict(test_x)

#Computing training and testing scores
train_score = balanced_accuracy_score(train_y, y_pred_train)
test_score = balanced_accuracy_score(test_y, y_pred_test)
print('Training score:', train_score)
print('Testing score:', test_score)

#%%
#Plot the confusion matrix
plot_confusion_matrix(svc, test_x, test_y)

#%%
#Prepare Submission
submission = pd.DataFrame()
submission['id'] = sample_sub['id']
predictions = svc.predict(data_test)
submission['y'] = predictions
submission.to_csv(r'C:\Users\fabia\Desktop\Studium\MSc\Semester 1\Advanced Machine Learning\Project 2\submission1_svc.csv')

