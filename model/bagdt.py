#! /nfs/raid13/babar/software/anaconda/bin/python
import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

#preprocess
pre = DataProcess(norm='l2', scale='std')
x_train = pre.transform(x_train)
x_test = pre.transform(x_test)

#training
H_bagdt = []
score_bagdt = []
print 'Start training bagdt...'
N_estimators = [int(1.5**x) for x in range(1,15)]
Max_depth=[1,2,3,4,5,7,10,20]
Min_samples_split=[2,3,4,5,7,10]

for i in range(len(Max_depth)):
	for j in range(len(N_estimators)):
		for k in range(len(Min_samples_split)):
			clf_base = DecisionTreeClassifier(max_depth=Max_depth[i],min_samples_split=Min_samples_split[k])
			clf = BaggingClassifier(base_estimator=clf_base, n_estimators=N_estimators[j])
			H_bagdt.append(clf)
			score_bagdt.append(cross_val_score(clf, x_train, y_train).mean())

print 'bagdt training done!'

if __name__ == '__main__':
        import csv
        print 'writing into file...'

        joblib.dump(H_bagdt, 'H_bagdt.pkl')
        with open('score_bagdt.txt','w') as f_sbagdt:
                b = csv.writer(f_sbagdt, delimiter='\n')
                b.writerow(score_bagdt)

        print 'writing done!'
