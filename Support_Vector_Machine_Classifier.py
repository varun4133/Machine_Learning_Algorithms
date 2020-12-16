import numpy as np



def loss(W, b, X_train, y_train, C):
    #C is l2 regularizer, lower C mean more regularization
    reg_loss=(W*W).sum()
    hinge_loss=C*((np.maximum(1-y_train*((W*X_train).sum(axis=1)+b),0))**2).sum()
    total_loss=reg_loss+hinge_loss
    return total_loss

def grad_loss(W, b, X_train, y_train, C):
    y_train=y_train.reshape(-1,1)
    y_hat=((W*X_train).sum(axis=1)+b).reshape(-1,1)
    wgrad=(2*W)+C*-2*((np.maximum(1-y_train*(y_hat),0))*y_train*X_train).sum(axis=0)
    
    bgrad=(C*-2*(np.maximum(1-y_train*(y_hat),0))*y_train).sum()
    
    return wgrad, bgrad

w, b, final_loss = minimize(objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000)