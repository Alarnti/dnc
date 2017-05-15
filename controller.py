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

import utils
import memory

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

LOGDIR = '/home/albert/Documents/ml/ML/ComputerVision/dnc/logs_controller/'

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
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

mem = memory.Memory(n_classes + size_ksi, n_classes,batch_size=batch_size)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes + size_ksi]))
    # 'after_lstm_out': tf.Variable(tf.random_normal([n_classes, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes + size_ksi]))
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


mem_vec = mem.make_request(pred)
print("mem_vec ", mem_vec.get_shape())

nn_pred = tf.matmul(pred,nn_out_w) + nn_out_b
print("nn_pred ", nn_pred.get_shape())

pred = nn_pred + mem_vec

pred = tf.squeeze(pred,[1])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print("pred after ", pred.get_shape())
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cost',cost)

merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
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




'''
import tensorflow as tf

import utils
import memory

import numpy as np
from random import shuffle
 
train_input = ['{0:020b}'.format(i) for i in range(2**10)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
 
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)



NUM_EXAMPLES = 500
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] 
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]



R = 1
N = 5
W = 21

size_ksi = W*R + 3 * W + 5 * R + 3

cols = 21

with tf.device("/cpu:0"):

    mem = memory.Memory()

    data = tf.placeholder("float", [1, 1,20]) #Number of examples, number of input, dimension of each input
    target = tf.placeholder("float", [1,21])
    num_hidden = 21 + size_ksi

    # initializer = tf.random_uniform_initializer(-0.01, 0.01)

    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden,state_is_tuple=True)
    # print cell.state_size
    # initial_state = cell.zero_state(1, tf.float32)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, data,dtype=tf.float32)

    # val, state = tf.contrib.rnn.static_rnn(lstm,state)#tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    # print val.get_shape()
    val = rnn_outputs
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
        
    #last = tf.reshape(last,[-1,num_hidden])

    last_copy_int = tf.identity(last)
    last_copy_int = tf.squeeze(last_copy_int)   

    last = tf.reshape(last_copy_int[:21],(1,21))

    interface = last_copy_int[22:]#tf.slice(last,22,-1)
    # print interface.get_shape()

    read_keys, read_strengths, key_write, write_srength, erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes = utils.transform_interface(interface, N, 20, R)
    mem_vec = mem.read_and_write(read_keys, read_strengths, key_write, write_srength, erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes)

    # print mem_vec.get_shape()


    weight = tf.Variable(tf.truncated_normal([num_hidden, cols]))# int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[cols])) #target.get_shape()[1]]))

    # print last.get_shape()

    # return_vec = 

    prediction = tf.nn.softmax(tf.matmul(weight,tf.transpose(last)) + bias)#tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)
    # mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    # error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    batch_size = 1000
    no_of_batches = int(len(train_input)) / batch_size
    epoch = 100#5000
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]

            ptr+=batch_size
            sess.run(minimize,feed_dict={data: inp[0], target: out[0]})
        print "Epoch ",str(i)
    #incorrect = sess.run(error,{data: test_input[0], target: test_output[0]})
    print sess.run(prediction,feed_dict={data: [[[1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0]]]})
    # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()
'''

# lstm_size = 


# lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# # Initial state of the LSTM memory.
# state = tf.zeros([batch_size, lstm.state_size])
# probabilities = []
# loss = 0.0
# for current_batch_of_words in words_in_dataset:
#     # The value of state is updated after processing each batch of words.
#     output, state = lstm(current_batch_of_words, state)

#     # The LSTM output can be used to make next word predictions
#     logits = tf.matmul(output, softmax_w) + softmax_b
#     probabilities.append(tf.nn.softmax(logits))
#     loss += loss_function(probabilities, target_words)





# class Controller:

# 	def __init__(input_size, output_size, read_heads_size, weighting_size):

# 		self.input_size = input_size
# 		self.output_size = output_size
# 		self.read_heads_size = read_heads_size
# 		self.weighting_size = weighting_size


# 		self.output_with_mem_size = self.weighting_size * self.read_heads_size + 3 * self.weighting_size + 5 * self.read_heads_size + 3



# 		data = tf.placeholder(tf.float32,[None,self.input_size,1])
# 		target = tf.placeholder(tf.float32,[None,self.output_with_mem_size])





