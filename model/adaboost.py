from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble

H1 = [] 
score1=[]
print 'Start training adaboost...'
Max_depth=[1, 2, 3, 4, 5, 7, 10, 20]
N_estimators=[int(1.5**p) for p in range(1,10)]
for i in range(len(Max_depth)):
    clf = tree.DecisionTreeClassifier(max_depth=Max_depth[i])
    for j in range(len(N_estimators)):
        dlf = ensemble.AdaBoostClassifier(base_estimator=clf, n_estimators=N_estimators[j])
        H1.append(dlf)
        score1.append(np.mean(cross_val_score(dlf, x_train, y_train)))              
print 'adaboost training done!'
