import numpy as np



def inner_product(X,Z=None):
   if Z is None:
         Z=X
   return X.dot(Z.T)
    
def L2_distance(X,Z=None):
    #finds distance between each vector in each matrix
    if Z is None:
        Z = X
    n,d1=X.shape
    m,d2=Z.shape
    x2=np.square(X).dot(np.ones((d1,m)))
    z2=np.square(Z).dot(np.ones((d2,n)))
    xz=inner_product(X,Z)
   
    distance=x2+(z2.T)-2*xz
    distance[distance<0]=0
    
    return np.sqrt(distance) 


def find_knn(X_train,X_test,k):
    #finds the top k neighbors given a training set and testing set
    #D is the actual distances while I is the indices of the neighbors in the training set
    #note that the output is k x m where m is number of training samples
    distance=L2_distance(X_train,X_test)
    D=np.sort(distance,axis=0)
    I=np.argsort(distance,axis=0)
    return (I[:k,:],D[:k,:])

def accuracy(y_true,y_pred):
    
    
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    e=y_true-y_pred
    acc=np.mean(e==0)
    return acc

def knnclassifier(X_train,y_train,X_test,k):
    #classifiest all test points based off training set and number of neighbors
    
    
    
    I,D=find_knn(X_train,X_test,k)
    neighbor_values=y_train[I]
    
    def mode(a):
        return(max(a,key=a.count))
    
    def modeC(array):
        return ([mode(list(column)) for column in array.T])
    
    predictions=np.array(modeC(neighbor_values))
    return predictions
    
    
    