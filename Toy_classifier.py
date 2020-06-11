#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np 
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (10.0,8.0) 
plt.rcParams['image.interpolation'] = 'nearest' 
plt.rcParams['image.cmap'] = 'gray' 


# In[109]:


np.random.seed(2) 
N = 100 
D = 2 
K = 20
X = np.zeros((N*K,D))
Y = np.zeros(N*K,dtype='uint8')

for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N)
    t = np.linspace(4*j,4*(j+1),N) + np.random.randn(N) * 0.2 
    X[ix] = np.c_[r*np.cos(t) , r*np.sin(t)]
    Y[ix] = j 
    fig = plt.figure() 
plt.scatter(X[:,0],X[:,1],s=60,c=Y,cmap=plt.cm.Spectral) 
plt.xlim([-1,1]) 
plt.ylim([-1,1]) 


# In[110]:


W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(200):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),Y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print( "iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),Y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db


# In[111]:


scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == Y)))


# In[112]:


h = 0.02 
xmin , xmax = X[:,0].min() - 1 , X[:,0].max() + 1 
ymin , ymax = X[:,1].min() - 1 , X[:,1].max() + 1 
xx , yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
u  = np.c_[xx.ravel(),yy.ravel()]
z = np.dot(u,W)+b 
z = np.argmax(z,axis=1)
z = z.reshape(xx.shape) 
fig = plt.figure() 
plt.contourf(xx,yy,z,cmap=plt.cm.Spectral,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=Y,s=44,cmap=plt.cm.Spectral) 
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())


# In[113]:


h = 100 
W = 0.01 * np.random.randn(D,h) 
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

step_size = 1e-0
reg = 1e-3 

num_examples = X.shape[0] 
for i in range(10000):
    hidden_layer = np.maximum(0,np.dot(X,W)+b)
    scores = np.dot(hidden_layer,W2)+b2
    exp_scores = np.exp(scores) 
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) 
    correct_logprobs = -np.log(probs[range(num_examples),Y])
    data_loss = np.sum(correct_logprobs)/num_examples
    loss = data_loss + reg_loss 
    if i%1000 == 0:
        print("Iteration %d : loss %f" %(i,loss))
    dscores = probs 
    dscores[range(num_examples),Y] -= 1 
    dscores/= num_examples 
    
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2


# In[114]:


hidden_layer = np.maximum(0,np.dot(X,W)+b) 
scores = np.dot(hidden_layer,W2) + b2 
predicted_class = np.argmax(scores,axis=1) 
print("Training accuracy = %0.2f" %(np.mean(predicted_class==Y)))


# In[115]:


# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_net.png')


# In[ ]:




