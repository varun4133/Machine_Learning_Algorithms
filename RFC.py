import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat




def sqimpurity(y_train):
    #Computes the weighted variance of the labels
    
    average=y_train.mean()
    impurity=((y_train-average)**2).sum()
    return impurity


def sqsplit(X_train, y_train):
    #Finds the best feature, cut value, and loss value.  
    bestloss = np.inf
    _,D = X_train.shape
    for col in range(D):
        xsort,idsort=np.unique(X_train[:,col],return_index=True)
        for p in range(len(xsort)-1):
            thresh=np.mean([xsort[p],xsort[p+1]])
            left=y_train[X_train[:,col]<thresh]
            right=y_train[X_train[:,col]>thresh]
            loss=(sqimpurity(left)+sqimpurity(right))
            if loss < bestloss:
                bestloss=loss
                feature=col
                cut=thresh
    return feature, cut, bestloss

class TreeNode:
    # Treenode class
    def __init__(self,left,right,feature,cut,prediction):
        self.left=left
        self.right=right
        self.feature=feature
        self.cut=cut
        self.prediction=prediction
        


    



def cart(X_train,y_train):
    #builds regression tree
    n,d=X_train.shape
    imp=sqimpurity(y_train)
    if (imp == 0) or (X_train==X_train[0]).all():
        prediction=y_train.mean()
        return TreeNode(None,None,None,None,prediction) 
    
    else:
        feature,cut,_=sqsplit(X_train,y_train)
        leftc=X_train[:,feature]<cut
        rightc=X_train[:,feature]>cut
        xleft,yleft=X_train[leftc],y_train[leftc]
        xright,yright=X_train[rightc],y_train[rightc]
        prediction=y_train.mean()
        return TreeNode(cart(xleft,yleft),cart(xright,yright),feature,cut,prediction)    
    
def evaltree(root,X_test):
   #gets predictions of Test set from built regression tree
    
   n,d=X_test.shape
   pred=np.zeros(n)
   def evaltree2(node,inds):
       if node.left==None:
           pred[inds]=node.prediction
       else:
            
           feature=node.feature
           cut=node.cut
           leftc=inds & (X_test[:,feature]<cut)
           rightc=inds & (X_test[:,feature]>=cut)
            
            
           evaltree2(node.left,leftc)
           evaltree2(node.right,rightc)
   starting_inds=np.ones(n,dtype=bool)
    
   evaltree2(root,starting_inds)
    
   return pred    


def spiraldata(N=300):
    #code to generate data
    r = np.linspace(1,2*np.pi,N) # generate a vector of "radius" values
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T # generate a curve that draws circles with increasing radius
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1])*0.2
    
    # Now sample alternating values to generate the test and train sets.
    xTe = xTr[::2,:]
    yTe = yTr[::2]
    xTr = xTr[1::2,:]
    yTr = yTr[1::2]
    
    return xTr, yTr, xTe, yTe

xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)
plt.scatter(xTrSpiral[:,0], xTrSpiral[:,1],30,c=yTrSpiral)
plt.colorbar()

def visclassifier(fun,xTr,yTr,w=None,b=0):
    yTr = np.array(yTr).flatten()
    

    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    plt.figure()

    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
            )
    
    if w is not None:
        w = np.array(w).flatten()
        alpha = -1 * b / (w ** 2).sum()
        plt.quiver(w[0] * alpha, w[1] * alpha,
            w[0], w[1], linewidth=2, color=[0,1,0])

    plt.axis('tight')
    # shows figure and blocks
    plt.show()

tree=cart(xTrSpiral,yTrSpiral) # compute tree on training data 
visclassifier(lambda X:evaltree(tree,X), xTrSpiral, yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(evaltree(tree,xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(evaltree(tree,xTeSpiral)) != yTeSpiral))

