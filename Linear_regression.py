

import tensorflow.compat.v1 as tf
print(tf.__version__)
tf.disable_v2_behavior()

import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation # imports animation support

# Generate random size of houses  in sq feet
num_houses = 160 
np.random.seed(42) 
house_size = np.random.randint(low=1000,high=3500,size=num_houses)

# Generate random prices of house in rupees 
np.random.seed(56) 
house_price =  house_size * 100.0 + np.random.randint(low=20000,high=70000,size=num_houses)
plt.plot(house_size , house_price , 'bx') 
plt.ylabel("Price") 
plt.xlabel("Size")
plt.show()

def normalize(array) : 
  return(array - array.mean())/(array.std())

# define the number of training and testing samples 
num_train_samples = math.floor(num_houses*0.7)

# define training data 

train_house_size = np.asarray(house_size[:num_train_samples]) 
train_house_price = np.asanyarray(house_price[:num_train_samples]) 

train_house_size_norm = normalize(train_house_size) 
train_house_price_norm = normalize(train_house_price)

# define test data 
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:]) 

test_house_size_norm = normalize(test_house_size) 
test_house_price_norm = normalize(test_house_price)

# Defining placeholders in tensorflow 

tf_house_size = tf.placeholder("float", name ="home_size")
tf_price = tf.placeholder("float", name="house_price")

# Defining variables for computation in tf 

tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor") 
tf_price_offset = tf.Variable(np.random.randn(), name = "price_offset")

# Once the tensors or the nodes of the DAG is defined we define the operations that occur : 
# during the prediction of the house prices 

tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size), tf_price_offset)

# Definition of the loss function to determine the error in the prediction 

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price ,2))/(2*num_train_samples)

# Definition of Learning rate 

learning_rate = 0.1

# Gradient descent optimizer 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initialization of the variables 
init = tf.global_variables_initializer() 

# Launching the graph in session 

with tf.Session() as sess : 
  sess.run(init) 

  # set how often to display the training data and number of training iterations
  display_every = 2.0 
  num_training_iter = 500 

  # Iteration of the training data 

  for iteration in range(num_training_iter): 

    # Fit all training data 
    for (x,y) in zip(train_house_price_norm,train_house_price_norm) : 
      sess.run(optimizer,feed_dict={tf_house_size:x, tf_price:y})

    # Display current status 
    if(iteration+1)% display_every == 0 :
      c = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_price:train_house_price_norm})
      print("iteration #:",'%04d' %(iteration+1) , "cost=","{:.9f}".format(c),\
            "size factor = " , sess.run(tf_size_factor), "price_offset = " ,sess.run(tf_price_offset))
    
  print("Optimization finished") 
  training_cost = sess.run(tf_cost,feed_dict={tf_house_size : train_house_size_norm, tf_price:train_house_price_norm})
  print("Trained cost = " ,training_cost , "size_factor = ",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))

