from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble

H2 = [] 
score2=[]
print 'Start training ada_rf...'
N_estimators = [10, 20, 50 ]
N_estimators1=[int(1.5**p) for p in range(3,9)]
Max_depth = [i+1 for i in range(9)]
for i in range(len(N_estimators)):
    for j in range(len(Max_depth)):
        for k in range(len(N_estimators1)):
            clf = ensemble.RandomForestClassifier(n_estimators=N_estimators[i], max_depth=Max_depth[j])
            dlf = ensemble.AdaBoostClassifier(base_estimator=clf, n_estimators=N_estimators1[k])
           
            H2.append(dlf)
            score2.append(np.mean(cross_val_score(dlf, x_train, y_train)))        
        

print 'ada_rf training done!'
