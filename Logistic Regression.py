
# coding: utf-8

# In[1]:

# import libraries 

import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
import scipy
from PIL import Image 
from scipy import ndimage 
from lr_utils import load_dataset


# In[4]:

# load dataset
trainset_x_og , trainset_y , testset_x_og , testset_y , classes = load_dataset()


# In[5]:


# preprocessing data

#index = 45 
#plt.imshow(trainset_x_og[index]) 
#print("y = " + str(trainset_y[:,index]) + "It is a " , classes[np.squeeze(trainset_y[:,index])].decode('utf-8') + " picture")

train_m = len(trainset_x_og)
test_m = len(testset_x_og)
num_pix = trainset_x_og[0]              # Setup structure 

temp1 = trainset_x_og.shape
temp2 = testset_x_og.shape              # Find shapes 

trainset_x_flat = trainset_x_og.reshape(trainset_x_og.shape[0],-1).T
testset_x_flat = testset_x_og.reshape(testset_x_og.shape[0],-1).T            # Reshape to problem

trainset_x = trainset_x_flat/255 
testset_x = testset_x_flat/255          # Normalize


# In[16]:


 # defining helper functions

def sigmoid(z):                           # inputs affine transformation
    s = 1.0 / 1.0 + np.ex(-1.0*x) 
    return s 

def initialize_with_zeros(dim):          # inputs the feature vector dimension
    w = np.zeros((dim,1))
    b = 0 
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b 

def forward_propagation(w,b,X,Y):        # inputs weights vector, bias , training set and training labels
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (np.sum(-(Y*np.log(A)+(1-Y)*np.log(1-A)))) * (1/m) 
    dw = (np.sum(np.dot(X,(A-Y).T))) *(1/m) 
    db = (np.sum(A-Y)) * (1/m) 
    cost = np.squeeze(cost)
    
    assert(dw.shape == w.shape) 
    assert(db.type == float)  
    assert(cost.shape == ())
    grads = {"dw":dw , "db":db}
    return grads,cost 

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):  
    costs=[]
    for iter in range(num_iterations):
        grads,cost = forward_propagation(w,b,X,Y)
        dw = grads["dw"] 
        db = grads["db"]
        w = w - dw 
        b = b - db 
        if i%100==0:
            costs.append(cost) 
        if print_cost & i%100 == 0: 
            print("Cost after &i iterations is %f",(i,cost))    
        params = {"w":w,"b":b}
        grad = {"dw":dw,"db":db}
        return params,grad,costs

def predict(w,X,Y):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b) 
    for i in range(A.shape[1]):
        if A[i]>=0.5:
            Y_prediction[i] = 1 
        else: 
            Y_prediction[i] = 0 
    assert(Y_prediction.shape == (1,m))
    return Y_prediction


# In[17]:


# defining model

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b = initialize_with_zeros(num_pix*num_pix*3)
    parameters,grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train) 
    
    print("Train accuracy", 100-np.mean(np.abs(Y_prediction_train-Y_train))*100)
    print("Test accuracy", 100-np.mean(np.abs(Y_prediction_test-Y_test))*100)
    
    d = {"costs":costs, "Y_prediction_test":Y_prediction_test , "Y_prediction_train":Y_prediction_train,"w":w,"b":b,"learning rate":learning_rate,"num_iterations":num_iterations}
    return d 


# In[19]:

d = model(trainset_x,trainset_y,testset_x,testset_y,num_iterations=2000,learning_rate=0.5,print_cost=True) 


# In[ ]:



