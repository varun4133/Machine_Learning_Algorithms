import numpy as np
import matplotlib.pyplot as plt
import time

#This creates a feed forward neural network that can be used for regression models
#can probably be used for classification too with a little tweaking

plot_relu=0

def ReLU (z):
    return np.maximum(z,0)

def ReLU_gradient(z):
    return (z>0).astype('float64')

if plot_relu==1:
    z_plot=np.linspace(-3,3,1000)
    
    fig=plt.figure()
    plt.plot(z_plot,ReLU(z_plot),'b-')
    plt.plot(z_plot,ReLU_gradient(z_plot),'c-')
    plt.xlabel('z')
    plt.ylabel('ReLU and Gradient')
    plt.legend(['ReLU','ReLU_gradient'])
    
def init_weights(specs):
    #len of specs is number of hidden layers (not including input and output) + 2
    #spec[0] must be dimension of input features and spec[-1] should be dimension of the output
    W=[]
    for i in range(len(specs)-1):
        W.append(np.random.randn(specs[i],specs[i+1]))
    return W

def forward_pass(W,X_train):
    
    
    #X_train is the input training set with a dimension of training samples x d
    #Output A is a list of matrices of length L+1 that contains the result of matrix multiplication at each layer
    #Output Z is a list of matrices of length L+1 that will contain the result of transition functions of A at each layer
    #Note: A[0] and Z[0] is merely the training set unchanged
    #A[-1]=Z[-1] because no transition function is used on the output layer
    #all of this is needed for backpropagation during training
    
    A=[X_train]
    Z=[X_train]
    
    L=len(W)
    
    for layer in range(L):
        a= (np.matmul(Z[layer],W[layer]))
        if layer < L-1:
            z=ReLU(a)
        else:
            z=a
        A.append(a)
        Z.append(z)
    return A,Z

def MSE(y_true,y_predictions):
    
    #This will be the cost function
    n=len(y_predictions)
    loss=((y_predictions-y_true)**2).sum()/n
    return loss

def MSE_grad(y_true,y_predictions):
    #This is the gradient of the cost function which will be used in backpropagation
    
    n=len(y_predictions)
    gradient=(2/n)*(y_predictions-y_true)
    return gradient

def backprop(W,A,Z,y_true):
    delta=(MSE_grad(y_true,Z[-1].flatten())*1).reshape(-1,1)
    L=len(W)
    gradients=[0]*L
    gradients[L-1]=delta

    for layer in range(L,1,-1):
        
        delta=ReLU_gradient(A[layer-1])*(np.matmul(gradients[layer-1],W[layer-1].T))
        gradients[layer-2]=delta
        gradients[layer-1]=np.matmul(gradients[layer-1].T,Z[layer-1]).T
    gradients[0]=np.matmul(gradients[0].T,Z[0]).T
    
    return gradients

def Training(X_train,y_train,learning_rate,epoch):
    #creates and train model with hidden layers given as input to init_weights
    #Note:y_train is a vector of length n and X_train is an array n by d
    #returns weights which can be used to predict using forward pass
    _,input_d=X_train.shape
    output_d=1
    
    losses=np.zeros(epoch)
    t0=time.time()
    
    
    W=init_weights([input_d,10,output_d])
    
    for i in range(epoch):
        
        A,Z= forward_pass(W,X_train)
        losses[i]=MSE(Z[-1].flatten(),y_train)
        
        
        gradients= backprop(W,A,Z,y_train)
        
        for j in range(len(W)):
            W[j]-=learning_rate*gradients[j]
    t1=time.time()
    
    print('Elasped time: %0.2fs' %(t1-t0))
    return W

def Predict(W,X_train):
    _,Z=forward_pass(W,X_train)
    return Z[-1]
    
    
X_train=np.array([[1,2,3],[0.5,5,3],[0.2,4,7]])
y_train=[2,4,10]

W=Training(X_train,y_train,0.001,30)
predictions=Predict(W,X_train)
   
    
        
    


    

     
    
