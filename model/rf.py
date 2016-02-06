#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

#preprocess
pre = DataProcess(norm='l2', scale='std')
x_train = pre.transform(x_train)
x_test = pre.transform(x_test)

#training
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


if __name__ == '__main__':
        import csv
        print 'writing into file...'

        joblib.dump(H_rf, 'H_rf.pkl')
        with open('score_rf.txt','w') as f_srf:
                b = csv.writer(f_srf, delimiter='\n')
                b.writerow(score_rf)

        print 'writing done!'



