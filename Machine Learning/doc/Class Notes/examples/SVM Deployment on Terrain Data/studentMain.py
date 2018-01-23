#!/usr/bin/python

""" Using the same strcuture of Naive Bayes exercice, run SVM classifier
    in the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


from sklearn import svm
from sklearn.metrics import accuracy_score

# Linear kernel - gamma does not change the output
clf_linear = svm.SVC(kernel='linear', gamma=1.0)
clf_linear.fit(features_train, labels_train)

pred = clf_linear.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_linear, features_test, labels_test)
print (' accurancy linear kernel: ', accuracy)

clf_linear_bigGamma = svm.SVC(kernel='linear', gamma=1000.0)
clf_linear_bigGamma.fit(features_train, labels_train)

pred = clf_linear_bigGamma.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_linear, features_test, labels_test)
print (' accurancy linear kernel and big gamma: ', accuracy)

# rbf kernel
clf_rbf = svm.SVC(kernel='rbf', gamma=1.0)
clf_rbf.fit(features_train, labels_train)

pred = clf_rbf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_rbf, features_test, labels_test)
print (' accurancy rbf kernel: ', accuracy)

clf_rbf_bigGamma = svm.SVC(kernel='rbf', gamma=1000.0)
clf_rbf_bigGamma.fit(features_train, labels_train)

pred = clf_rbf_bigGamma.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_rbf_bigGamma, features_test, labels_test)
print (' accurancy rbf kernel and big gamma: ', accuracy)

# For Quiz 22 - Does a large C mean you expect a smooth boundary, or
# you will get more training points correct ?
# Large C should get more points correct but when printing the 
# accurancy for C=1 and C=1000. C=1 has a better one if gamma is big.
clf_rbf_bigGamma_bigC = svm.SVC(kernel='rbf', gamma=1000.0, C=1000.0)
clf_rbf_bigGamma_bigC.fit(features_train, labels_train)

pred = clf_rbf_bigGamma_bigC.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_rbf_bigGamma_bigC, features_test, labels_test)
print (' accurancy rbf kernel and big gamma, big C: ', accuracy)

# This kernel configuration has the best accurancy
clf_rbf_bigC = svm.SVC(kernel='rbf', gamma=1.0, C=1000.0)
clf_rbf_bigC.fit(features_train, labels_train)

pred = clf_rbf_bigC.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
prettyPicture(clf_rbf_bigC, features_test, labels_test)
print (' accurancy rbf kernel and big C: ', accuracy)