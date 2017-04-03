import tensorflow as tf

def oneplus(beta):
	return 1 + tf.log(1 + tf.exp(beta))



def transform_interface(interface, N, W, R):
	border_index = 0

	read_keys = tf.reshape(interface[border_index:R*W],[R,W])
	#print read_keys.get_shape()
	border_index += R*W



	read_strengths = oneplus(tf.reshape(interface[border_index:border_index + R],[-1]))
	border_index += R

	print 'strengths ', read_strengths.get_shape()

	key_write = tf.reshape(interface[border_index: border_index + W],[-1])
	border_index += W

	write_srength = oneplus(tf.reshape(interface[border_index],[]))
	border_index += 1

	erase_vec = tf.sigmoid(tf.reshape(interface[border_index:border_index + W],[1,W]))
	border_index += W

	write_vec = tf.reshape(interface[border_index:border_index + W],[1,W])
	border_index += W

	free_gates = tf.sigmoid(tf.reshape(interface[border_index:border_index + R ],[-1]))
	border_index += R

	allocation_gate = tf.sigmoid(tf.reshape(interface[border_index],[]))
	border_index += 1

	write_gate = tf.sigmoid(tf.reshape(interface[border_index],[]))
	border_index += 1

	read_modes = tf.sigmoid(tf.reshape(interface[border_index:border_index + 3],[-1]))
	border_index += 3

	return read_keys, read_strengths, key_write, write_srength, erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes



