

import tensorflow as tf
print(tf.__version__) 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIT_data/",one_hot=True)

# Initialize place holders to contain the intensity of brightness of the input images. 
# Initialize placeholders to contain the probability that the digit in the image corresponds to one of the 10 digits used in the database 

x = tf.placeholder(tf.float32,shape=[None,784]) 
y_ = tf.placeholder(tf.float32, shape=[None,10])

# Definition of the model : Using softmax activation function : returns probability distribution for every value of last layer neuron
W = tf.Variable(tf.zeros([784,10])) 
b = tf.Variable(tf.zeros([10]))

# Softmax activation function : 
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss measurement 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ , logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer() 

with tf.Session() as sess : 
  sess.run(init) 
  for i in range(1000): 
    batch_xs , batch_ys = mnist.train.next_batch(100)   # get 100 random data points from the data.batch+xs = image
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys}) 
  
  # Test for correct prediction 
  correct_prediction = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

  test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})
  print("Test Accuracy : {0}%".format(test_accuracy*100.0))

  sess.close()

