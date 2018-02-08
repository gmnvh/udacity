#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)

### Accuracy
from sklearn.metrics import accuracy_score

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print 'Accuracy: ', accuracy


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

### For L4Q12 - What is the accuracy for min_spamples_split 2 and 50 ?

from sklearn import tree

clf_2 = tree.DecisionTreeClassifier()
clf_50 = tree.DecisionTreeClassifier(min_samples_split=50)

clf_2 = clf_2.fit(features_train, labels_train)
clf_50 = clf_50.fit(features_train, labels_train)

pred_2 = clf_2.predict(features_test)
pred_50 = clf_50.predict(features_test)

from sklearn.metrics import accuracy_score

acc_min_samples_split_2 = accuracy_score(pred_2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred_50, labels_test)

print 'Accuracy min_samples_split_2 :', acc_min_samples_split_2
print 'Accuracy min_samples_split_50 :', acc_min_samples_split_50