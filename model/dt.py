import sys
sys.path.append('/home/yunxuanli/kagglecs155/ml155')


from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

H_dt = []
score_dt = []
print 'Start training DT candidates...'
Max_depth=[1,2,3,4,5,7,10,20]
Min_samples_split=[2,3,4,5,7,10]
for i in range(len(Max_depth)):
	for j in range(len(Min_samples_split)):
		clf = DecisionTreeClassifier(max_depth=Max_depth[i], min_samples_split=Min_samples_split[j])
		score_dt.append(cross_val_score(clf, x_train, y_train).mean())
		H_dt.append(clf)

print 'Decision Tree training Done\n'

if __name__ == '__main__':
	import csv
	print 'writing into file...'

	with open('H_dt.txt','w') as f_hdt:
		a = csv.writer(f_hdt, delimiter='\n')
		a.writerow(H_dt)
	with open('score_dt.txt','w') as f_sdt:
		b = csv.writer(f_sdt, delimiter='\n')
		b.writerow(score_dt)
	
	print 'writing done!'
