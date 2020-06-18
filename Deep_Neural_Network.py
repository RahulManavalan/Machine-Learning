
# coding: utf-8

# In[3]:

import numpy as np 
import matplotlib.pyplot as plt
import h5py 

def sigmoid(Z):
    A = 1 / (1+np.exp(-Z))
    cache = Z 
    return A , cache 

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A,cache 

def relu_backward(dA,cache):
    Z = cache 
    dZ = np.array(dA,copy=True) 
    return dZ 

def sigmoid_backward(dA,cache):
    Z = cache 
    s = 1 / (1+np.exp(-Z))
    dZ = dA*s*(1-s)
    return dZ 

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['test_set_y'][:])
    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    classes = np.array(test_dataset['list_classes'][:]) 
    train_set_y_orig = train_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,train_set_y_orig,classes

def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters={} 
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01 
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
        return parameters 

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev)+b 
    cache = (A,W,b) 
    return Z,cache 

def linear_activation_forward(A_prev,W,b,activation):
    if activation=='sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z) 
    if activation=='relu':
        Z,linear_activation=linear_forward(A_prev,W,b) 
        A,activation_cache = relu(Z) 
    cache = (linear_cache,activation_cache) 
    return A,cache 

def L_model_forward(X,parameters):
    caches=[]
    A=X
    L = len(parameters)//2 
    
    for l in range(1,L):
        A_prev = A 
        A ,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],
                                            parameters['b'+str(l)],activation='relu')
        caches.append(cache) 
    AL ,cache = linear_activation_forward(A_prev,parameters['W'+str(L)],
                                         parametes['b'+str(L)],activation='sigmoid')
    return AL,caches 

def compute_cost(AL,Y):
    m = Y.shape[1] 
    cost = (1/m)*(np.sum(-(Y*np.log(AL)+(1-Y)*np.log(1-AL))))
    cost = np.squeeze(cost) 
    return cost 

def linear_backward(dZ,cache):
    
    A_prev ,W ,b = cache 
    m = A_prev.shape[1] 
    dW = (1/m)*np.dot(dZ,A_prev.T) 
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True) 
    dA_prev = np.dot(W.T,dZ) 
    return dA_prev,dW,db 

def linear_activation_backward(dA,cache,activation):
    linear_cache , activation_cache = cache
    if activation=='relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache) 
    if activation=='sigmoid':
        dZ = sigmoid_backward(dA,activation_cache) 
        dA_prev,dW,db = linear_backward(dZ,linear_cache) 
    return dA_prev,dW,db 

def L_model_backward(AL,Y,caches):
    grads={} 
    L = len(caches) 
    m = AL.shape[1] 
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    current_cache = caches[L-1] 
    grads['dA'+str(L-1)] , grads['dW'+str(L)] , grads['db'+str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_temp ,dW_temp , db_temp = linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')
        grads['dA'+str(l)] = dA_temp 
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
    return grads 

def update_parameters(paramters,grads,learning_rate):
    L = len(parameters)//2 
    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate*grads['db'+str(l+1)]
    return parameters

def predict(X,y,parameters):
    m = X.shape[1] 
    n = len(parameters) // 2  
    p = np.zeros((1,m))
    probas,caches = L_model_forward(X,parameters)
    
    for i in range(0,probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1 
        else : 
            p[0,i] = 0 
    print("Accuracy:"+str(np.sum((p==y)/m)))
    return p 

def print_mislabeled_images(clasees,X,y,p):
    a = p+y 
    mislabeled_indices = np.asaaray(np.where(a==1))
    plt.rcParams['figure.figsize']=(40.0,40.0) 
    num_images = len(mislabeled_indices[0]) 
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2,num_images,i+1) 
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


# In[5]:

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)


train_x_orig,train_y  ,test_x_orig , test_y,classes = load_data() 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255


# In[8]:

layers_dims = [12288, 20, 7, 5, 1]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X,parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL,Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL,Y,caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters,grads,learning_rate) 
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[ ]:

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[ ]:



