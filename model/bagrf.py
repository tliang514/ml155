#! /nfs/raid13/babar/software/anaconda/bin/python
import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

#preprocess
pre = DataProcess(norm='l2', scale='std')
x_train = pre.transform(x_train)
x_test = pre.transform(x_test)

#training
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

if __name__ == '__main__':
        import csv
        print 'writing into file...'

        joblib.dump(H_bagrf, 'H_bagrf.pkl')
        with open('score_bagrf.txt','w') as f_sbagrf:
                b = csv.writer(f_sbagrf, delimiter='\n')
                b.writerow(score_bagrf)

        print 'writing done!'
