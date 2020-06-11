
# coding: utf-8

# In[27]:

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn 
from testCases_v2 import * 
import sklearn.datasets 
import sklearn.linear_model
from planar_utils import sigmoid , load_planar_dataset , load_extra_datasets
from planar_utils import plot_decision_boundary
get_ipython().magic('matplotlib inline')
np.random.seed(1) 


# In[24]:

X,Y = load_planar_dataset() 


# In[25]:

plt.scatter(X[0,:],X[1,:],s=40,c=Y,cmap=plt.cm.Spectral)
print(X.shape , 
Y.shape)


# In[43]:

clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X.T,Y.T)

plot_decision_boundary(lambda x:clf.predict(x) , X,Y)
plt.title("Logistic Regression")
LR_prediction = clf.predict(X.T) 
Accuracy = (np.dot(Y,LR_prediction) + np.dot((1-Y),(1-LR_prediction)))/float(Y.size)
print("Accuracy of Logistic Regression= ", Accuracy*100)


# In[ ]:

def layer_sizes(X,Y):
    nx = X.shape[0] 
    nh = 4 
    ny = Y.shape[0]
    return nx,nh,ny

def initialize_parameters(nx,nh,ny):
    W1 = np.random.randn(nh,nx) * 0.01 
    b1 = np.zeros((nh,1))
    W2 = np.random.randn(ny,nh) * 0.01 
    b2 = np.zeros((ny,1))
    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2} 
    return parameters 

def foraward_propagation(X,parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1 
    A1 = np.tanh(Z1) 
    Z2 = np.dot(W2,A1) + b2 
    A2 = sigmoid(Z2)
    
    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2,cache

def compute_cost(Y,A2):  
    m = Y.shape[1]
    cost = (np.sum(-(Y*np.log(A2) + (1-Y)*np.log(1-A2))))*(1/m)
    cost = float(np.squeeze(cost))
    assert(isinstance(cost,float))
    return cost 

def back_propagation(parameters,cache,X,Y):
    
    A2 = cache["A2"] 
    A1 = cache["A1"] 
    W1 = parameters["W1"]
    W2 = parameters["W2"] 
    m = X.shape[1] 
    
    dZ2 = A2 - Y 
    dW2 = (np.dot(dZ2,A1.T)) * (1/m)
    db2 = np.sum(dZ2,axis=1,keepdims=True) * (1/m) 
    dZ1 = np.multiply(np.dot(W2.T,dZ2) , (1-np.power(A1,2)))
    dW1 = (np.dot(dZ1,X.T))*(1/m) 
    db2 = np.sum(dZ1,axis=1,keepdims=True) *(1/m)
    
    grads = {"dW1":dW1 , "dW2":dW2 , "db1":db1 , "db2":db2}
    return grads 

def update_parameters(grads,parameters,learning_rate=1.12): 
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1 
    W2 = W2 - learning_rate*dW2 
    b1 = b1 - learning_rate*db1 
    b2 = b2 - learning_rate*db2 
    
    parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

    return parameters

def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters) 
    predictions = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        if A2[0][i] >= 0.5: predictions[0][i] = 1 
        else: predictions[0][i] = 0 
    return predictions 


# In[ ]:

def nn_model(X,Y,nh,num_iterations,print_cost=False):
    np.random.seed(2) 
    nx = layer_sizes(X,Y)[0]
    ny = layer_sizes(X,Y)[2]
    parameters = initialize_parameters(nx,nh,ny) 
    
    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters) 
        cost = compute_cost(Y,A2) 
        grads = back_propagation(parameters,cache,X,Y) 
        parameters = update_parameters(grads,parameters) 
        if print_cost and i%100 == 0 : 
            print("Cost after iteration %i = %f" %(i,cost)) 
            
    return parameters 


# In[ ]:

plt.figure(figsize=(16,32)) 
hidden_layers = [1,2,3,4,5,20,50] 
for i,nh in enumerate(hidden_layers): 
    plt.subplot(5,2,i+1)
    plt.title("Hidden layer of size %d" %nh) 
    parameters = nn_model(X,Y,nh,num_iterations=5000,print_cost=False) 
    plot_decision_boundary(lambda x:predict(parameters,x.T) , X, Y) 
    predictions = predict(parameters,X)
    accuracy = (float(np.dot(Y,predictions.T) + np.dot(1-Y,(1-predictions).T))/(float(Y.size)))*100
    print("Accuracy for %d hidden layers = %f" %(nh,accuracy)) 

