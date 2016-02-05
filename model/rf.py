import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

H_rf = []
score_rf = []
print 'Start training RF candidates...'
N_estimators = [10, 20, 50]
Max_depth = [i+1 for i in range(10)]
for i in range(len(N_estimators)):
	for j in range(len(Max_depth)):
		clf = RandomForestClassifier(n_estimators=N_estimators[i], max_depth=Max_depth[j])
		score_rf.append(cross_val_score(clf, x_train, y_train).mean())
		H_rf.append(clf)

print 'RF training Done\n'
