
# coding: utf-8

# In[13]:

import numpy as np 
import h5py 
import matplotlib.pyplot as plt
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (5.0,4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'grey'
get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
np.random.seed(1) 


# In[7]:

def initialize_parameters_deep(layer_dims):
    np.random.seed(3) 
    parameters = {} 
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l],0))
    
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b 
    cache = (A,W,b) 
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation=='sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b) 
        A,activation_cache = sigmoid(Z) 
    if activation=='relu':
        Z,linear_cache = linear_forward(A_prev,W,b) 
        A,activation_cache = relu(Z)
    caches = (linear_cache,activation_cache)
    
    return A,caches

def L_model_forward(X,parameters):
    cachoes = [] 
    A = X 
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A 
        A , caches = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        cachoes.append(caches) 
    AL , caches = linear_activation_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    cachoes.append(caches) 
    
    return AL,cachoes

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (1/m)*np.sum(-(Y*np.log(AL) + (1-Y)*np.log(1-AL)))
    cost = np.squeeze(cost) 
    
    return cost

def linear_backward(dZ,linear_cache):
    A_prev,W,b = cache 
    m = A_prev.shape[1] 
    dW = (1/m)*(np.dot(dZ,A_prev.T))
    db = (1/m)*(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ) 
    
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache , activation_cache = cachoes
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache) 
        dA_prev,dW,db = linear_backward(dZ , linear_cache) 
    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache) 
        
    return dA_prev,dW,db

def L_model_backward(AL,Y,cachoes):
    grads={}
    L = len(cachoes)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = -(np.divide(Y,AL) - np.divide((1-Y),(1-AL)))
    current_cache = cachoes[L-1]
    grads['dA'+str(L-1)] , grads['dW'+str(L-1)] , grads['db'+str(L-1)] = linear_activation_backward(dAL,current_cache,'sigmoid')
    for i in reversed(range(L-1)):
        current_cache = cachoes[l]
        dA_prev_temp , dW_temp , db_temp = linear_activation_backward(dAL,current_cache,'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
        
    return grads

def update_parameters(paramters,grads,learning_rate):
    L = len(parameters)//2 
    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate*grads['db'+str(l+1)]
    
    return parameters


# In[18]:




# In[19]:




# In[ ]:



