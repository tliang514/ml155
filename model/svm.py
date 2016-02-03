import sys
sys.path.append('/Users/yunxuanli/Documents/Caltechphd/courses/MachineLearning/sentimentanalysis/code/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import svm

H = []
score = []
print 'Start training SVM candidates...'

c = [float(10**x) for x in range(-3,3)]
Gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 2.]
for i in range(len(c)):
	for j in range(len(Gamma)):
		clf = svm.SVC(C=c[i], gamma=Gamma[j], kernel='rbf')
		score.append(cross_val_score(clf, x_train, y_train).mean())
		H.append(clf)
print 'SVM training Done\n'

