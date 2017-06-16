'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import sys
sys.path.append("../")

import memory

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

LOGDIR = 'logs_controller/'

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

R = 1
N = 5
W = 10

size_ksi = W*R + 3 * W + 5 * R + 3


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

mem = memory.Memory(n_classes + size_ksi, n_classes,R=R,W=W,N=N,batch_size=batch_size)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([ n_hidden, n_classes + size_ksi]),'after_rnn_weights')
    # 'after_lstm_out': tf.Variable(tf.random_normal([n_classes, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([ n_classes + size_ksi]), 'after_rnn_bias')
}



def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

pred = tf.expand_dims(pred,1)

print("pred ", pred.get_shape())

nn_out_w = tf.Variable(tf.truncated_normal([batch_size, n_classes + size_ksi,n_classes],stddev=0.1))
nn_out_b = tf.Variable(tf.fill([batch_size, 1,n_classes],0.2))

# pred = tf.expand_dims(pred,[1])
# print("mem_vec before ", pred.get_shape())
# mem_vec = mem.make_request(pred)#tf.map_fn(lambda x: mem.make_request(x), pred,back_prop=True)#
# print("mem_vec ", mem_vec.get_shape())

# pred = tf.squeeze(pred,[1])
nn_pred = tf.matmul(pred,nn_out_w) + nn_out_b
print("nn_pred ", nn_pred.get_shape())

pred = nn_pred #+ mem_vec

# pred = tf.squeeze(pred,[1])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print("pred after ", pred.get_shape())
pred = tf.squeeze(pred,[1])
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cost',cost)
tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)
    writer = tf.summary.FileWriter(LOGDIR + 'contr')
    writer.add_graph(sess.graph)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        # print('--------- M ----------')
        # print(sess.run(mem.M, feed_dict={x: batch_x, y: batch_y}))
        # print('--------- M before----------')
        # print(sess.run(mem.M_before, feed_dict={x: batch_x, y: batch_y}))
        # print('-------Read Weighting----------')
        # print(sess.run(mem.read_weighting, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- L ------')
        # print(sess.run(mem.L, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- read keys -----')

        # print(sess.run(mem.read_keys, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- read strengths -------')
        # print(sess.run(mem.read_strengths, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- write_key --------')
        # print(sess.run(mem.write_key, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- write strength -----')
        # print(sess.run(mem.write_strength, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- erase vec --------')
        # print(sess.run(mem.erase_vec, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- write vec --------')
        # print(sess.run(mem.write_vec, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- free gates --------')
        # print(sess.run(mem.free_gates, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- allocation gate ------')
        # print(sess.run(mem.allocation_gate, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- write gate --------')
        # print(sess.run(mem.write_gate, feed_dict={x: batch_x, y: batch_y}))
        # print('-------- read modes --------')
        # print(sess.run(mem.read_modes, feed_dict={x: batch_x, y: batch_y}))
        # print('----- mem_out -----')
        # print(sess.run(mem_vec, feed_dict={x: batch_x, y: batch_y}))
        # print('----- c --------')
        # print(sess.run(mem.c,feed_dict={x: batch_x, y: batch_y}))
        # print('*************************')
        merg, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})

        writer.add_summary(merg,step)

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1     
    print("Optimization Finished!")

    writer.close()
    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
