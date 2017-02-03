import tensorflow as tf

import numpy as np
from random import shuffle
 
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
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



NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] 
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

data = tf.placeholder(tf.float32,[None,20,1])

v_vector = tf.placeholder(tf.float32,[None,21,1])
interface_vector = tf.placeholder(tf.float32,[None,30,1])

target = tf.placeholder(tf.float32,[None,21])

num_hidden = 10
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)	

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

#first_score = tf.matmul(last, weight) + bias #tf.nn.softmax(tf.matmul(last, weight) + bias)

weight_v = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias_v = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

weight_interface = tf.Variable(tf.truncated_normal([num_hidden, int(interface_vector.get_shape()[1])]))
bias_interface = tf.Variable(tf.constant(0.1, shape=[interface_vector.get_shape()[1]]))

cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 100
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch - ",str(i)
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()



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





class Controller:

	def __init__(input_size, output_size, read_heads_size, weighting_size):

		self.input_size = input_size
		self.output_size = output_size
		self.read_heads_size = read_heads_size
		self.weighting_size = weighting_size


		self.output_with_mem_size = self.weighting_size * self.read_heads_size + 3 * self.weighting_size + 5 * self.read_heads_size + 3



		data = tf.placeholder(tf.float32,[None,self.input_size,1])
		target = tf.placeholder(tf.float32,[None,self.output_with_mem_size])





