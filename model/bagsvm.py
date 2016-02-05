import sys
sys.path.append('/Users/yunxuanli/Documents/Caltechphd/courses/MachineLearning/sentimentanalysis/code/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

H_bagsvm = []
score_bagsvm = []
print 'Start training bagsvm...'
Max_samples = [1, 2, 3, 5, 7]
N_estimators = [int(1.5**x) for x in range(1,15)]
c = [float(10**x) for x in range(-3, 4)]
for i in range(len(Max_samples)):
	for j in range(len(N_estimators)):
		for k in range(len(c)):
			clf_base = svm.SVC(C=c[k])
			clf = BaggingClassifier(base_estimator=clf_base, n_estimators=N_estimators[j], max_samples=Max_samples[i])
			H_bagsvm.append(clf)
			score_bagsvm.append(cross_val_score(clf, x_train, y_train).mean())

print 'bagsvm training done!'
