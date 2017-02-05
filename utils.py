import tensorflow as tf

def oneplus(beta):
	return 1 + tf.log(1 + tf.exp(beta))



def transform_interface(interface, N, W, R):
	border_index = 0

	read_keys = tf.reshape(interface[border_index:R*W - 1],[R,W])
	border_index += R*W

	read_strengths = oneplus(interface[border_index:border_index + R - 1])
	border_index += R

	key_write = interface[border_index: border_index + W - 1]
	border_index += W

	write_srength = oneplus(interface[border_index])
	border_index += 1

	erase_vec = tf.sigmoid(interface[border_index:border_index + W - 1])
	border_index += W

	write_vec = interface[border_index:border_index + W - 1]
	border_index += W

	free_gates = oneplus(interface[border_index:border_index + R - 1])
	border_index += R

	allocation_gate = interface[border_index]
	border_index += 1

	write_gate = interface[border_index]
	border_index += 1

	read_modes = interface[border_index:border_index + 3 - 1]
	border_index += 3

	return read_keys, read_strengths, key_write, write_srength, erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes



