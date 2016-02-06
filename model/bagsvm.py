#! /nfs/raid13/babar/software/anaconda/bin/python
import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')

from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

#preprocess
pre = DataProcess(norm='l2', scale='std')
x_train = pre.transform(x_train)
x_test = pre.transform(x_test)

#training
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

if __name__ == '__main__':
        import csv
        print 'writing into file...'

        joblib.dump(H_bagsvm, 'H_bagsvm.pkl')
        with open('score_bagsvm.txt','w') as f_sbagsvm:
                b = csv.writer(f_sbagsvm, delimiter='\n')
                b.writerow(score_bagsvm)

        print 'writing done!'
