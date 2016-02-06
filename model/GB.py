from data.DataPreprocess import *
from data.DataProcess import DataProcess

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble

H3 = [] 
score3=[]
print 'Start training GB...'
N_estimators = [100, 200, 300, 500 ]
Max_depth = [i+1 for i in range(14)]

for i in range(len(N_estimators)):
    for j in range(len(Max_depth)):
      clf = ensemble.GradientBoostingClassifier(n_estimators=N_estimators[i], max_depth=Max_depth[j])
      H3.append(clf)
      score3.append(np.mean(cross_val_score(clf, x_train, y_train)))  
print 'GB training done!'
