import sys
sys.path.append('/Users/yunxuanli/Documents/Caltechphd/courses/MachineLearning/sentimentanalysis/code/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

H_bagrf = []
score_bagrf = []
print 'Start training bagrf...'
N_estimators1 = [10, 20, 50]
N_estimators2 = [int(1.5**x) for x in range(1,15)]
Max_depth = [i+1 for i in range(10)]

for i in range(len(N_estimators1)):
	for j in range(len(N_estimators2)):
		for k in range(len(Max_depth)):
			clf_base = RandomForestClassifier(n_estimators=N_estimators1[i], max_depth=Max_depth[k])
			clf = BaggingClassifier(base_estimator=clf_base, n_estimators=N_estimators2[j])
			H_bagrf.append(clf)
			score_bagrf.append(cross_val_score(clf, x_train, y_train).mean())

print 'bagrf training done!'
