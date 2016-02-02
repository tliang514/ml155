#Author: Yunxuan
#Date: 02/02/2016

# coding: utf-8
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer 

class DataProcess:
    #class used to preprocess data
    """
    Parameters
    -----------
    norm: 'l1', 'l2' or 'None'
          Normalization methods.
                  
    scale: 'std', 'MinMax' or 'None'
           Scale methods.
           
    Attributes
    ----------
    transform: 
    tfidf:
    
    """
    
    def __init__(self, norm='l2', scale='std'):
        self.norm = norm
        self.scale = scale
        
    def transform(self, X):
        """Norm and Scale the rows of X
        
        Input:
            X: numpy.ndarray. training data features
                    
        Output:
            The transformed numpy.ndarray.    
        """
        
        if self.norm not in ('l1', 'l2', 'max', None):
            raise ValueError("'%s' is not a supported norm method" %self.norm)
        elif self.norm==None:
            normer = X
        else:
            normer = normalize(X, norm=self.norm, copy=True)
        
        if self.scale not in ('std', None, 'minmax'):
            raise ValueError("'%s' is not a supported scale method" %self.scale)
        elif self.scale=='std':
            scaler = scale(normer, axis=0, with_mean=True, with_std=True, copy=True)
        elif self.scale==None:
            scaler = normer
        elif self.scale=='minmax':
            scaler = MinMaxScaler().fit_transform(normer)
        
        return scaler
    
    def tfidf(self, X):
        """TF-IDF transform of X
        
        Input:
            X: numpy.ndarray. training data features
            
        Output:
            The TF-IDF transformed numpy.ndarray.
        """
        
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        return tfidf.toarray()



#Test
if __name__ == '__main__':
	from DataProcess import DataProcess
	import numpy as np
	x = np.array([[1.0,2,3],[4.1,5,6]])
	print x

	print 'Test norm'
	clf = DataProcess(norm='l2', scale=None)
	print clf.transform(x)

	print 'Test std scale'
	clf = DataProcess(norm=None, scale='std')
	print clf.transform(x)

	print 'Test minmax scale'
	clf = DataProcess(norm=None, scale='minmax')
	print clf.transform(x)

	print 'Test tfidf'
	print clf.tfidf(x)
