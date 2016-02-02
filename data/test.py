from DataProcess import DataProcess
import numpy as np
x = np.array([[1.0,2,3],[4.1,5,6]])
print x

print 'Test norm'
clf = DataProcess(norm='l2',scale=None)
print clf.transform(x)

print 'Test std scale'
clf = DataProcess(norm=None, scale='std')
print clf.transform(x)

print 'Test minmax scale'
clf = DataProcess(norm=None, scale='minmax')
print clf.transform(x)

print 'Test tfidf'
print clf.tfidf(x)

