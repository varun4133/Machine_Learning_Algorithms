import numpy as np
import matplotlib.pyplot as plt

#Important note: Here, training points must be -1 or 1 instead of 0 and 1

def sigmoid(z):
    
    sgmd= np.reciprocal(1+np.exp(-1*z))
    
    return sgmd

def log_loss(X_train, y_train, W, b=0):
    #calculates binary cross entropy
    #note that y values are either 1 or -1
    
    binary_cross_entropy=-1*np.sum(np.log(sigmoid(y_train*(X_train.dot(W)+b))))   
    
    return binary_cross_entropy

def loss_gradient(X_train, y_train, W, b):
    wgrad=np.sum((-y_train*sigmoid(-y_train*(X_train.dot(W)+b))*X_train.T).T,axis=0)
    bgrad=np.sum(-y_train*sigmoid(-y_train*(X_train.dot(W)+b)))
    return wgrad, bgrad

def logistic_regression_gradient_descent(X_train, y_train, max_iter, learning_rate):
    _, d = X_train.shape
    W = np.zeros(d)
    b = 0.0
    
    
    for _ in range(max_iter):
        wgrad,bgrad=loss_gradient(X_train,y_train,W,b)
        W=W-learning_rate*wgrad
        b=b-learning_rate*bgrad
       
        
        
    return W, b


def Predict(X_test, W, b=0):
    #returns probablity and prediction of test points
    prob=sigmoid(X_test.dot(W)+b)
    
    prediction = (prob > 0.5).astype(int)
    prediction[prediction != 1] = -1

    return prediction,prob

#testing model out

n_samples=500
class_one = np.random.multivariate_normal([5, 10], [[1, .25],[.25, 1]], n_samples)
class_one_labels = -np.ones(n_samples)

class_two = np.random.multivariate_normal([0, 5], [[1, .25],[.25, 1]], n_samples)
class_two_labels = np.ones(n_samples)

features = np.vstack((class_one, class_two))
labels = np.hstack((class_one_labels, class_two_labels))


max_iter = 10000
alpha = 1e-4
final_w, final_b = logistic_regression_gradient_descent(features, labels, max_iter, alpha)

scores,_ = Predict(features, final_w, final_b)

pred_labels = (scores > 0.5).astype(int)
pred_labels[pred_labels != 1] = -1

plt.figure(figsize=(9, 6))

# plot the decision boundary 
x = np.linspace(np.amin(features[:, 0]), np.amax(features[:, 0]), 10)
y = -(final_w[0] * x + final_b)/ final_w[1] 
plt.plot(x, y)

plt.scatter(features[:, 0], features[:, 1],
            c = pred_labels, alpha = .6)
plt.title("Predicted labels", size=15)
plt.xlabel("Feature 1", size=13)
plt.ylabel("Feature 2", size=13)
plt.axis([-3,10,0,15])


    
    




