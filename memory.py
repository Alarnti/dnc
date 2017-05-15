import tensorflow as tf
import numpy as np

class Memory:

	def __init__(self, nn_size, output_size, batch_size=1, R=1, N=5, W=15):
		with tf.variable_scope('memory'):

			self.batch_size = batch_size

			self.R = R
			self.N = N
			self.W = W
			self.M = tf.Variable(tf.fill([self.batch_size, self.N,self.W],1e-6))
			self.L = tf.Variable(tf.fill([self.batch_size, self.N,self.N],1e-6))

			self.psi = tf.Variable(tf.zeros([1,self.N]))
			self.u = tf.Variable(tf.zeros([self.batch_size,self.N]))
			self.free_list = tf.Variable(tf.zeros([1,self.N]))

			self.write_weighting = tf.Variable(tf.fill((self.batch_size,self.N),1e-6))
			# self.last_write_weighting = tf.Variable(tf.fill((1,self.N),1e-7))

			self.read_weighting = tf.Variable(tf.fill([self.batch_size,self.R,self.N],1e-6))
			#self.last_read_weighting = tf.Variable(tf.zeros([self.R, self.N]))

			self.read_vecs = tf.Variable(tf.fill([self.batch_size, self.R,self.W],1e-6))

			self.p = tf.Variable(tf.fill((self.batch_size,self.N),1e-6))
			self.forward = 0
			self.backward = 0

			self.interface_size = self.W*self.R + 3 * self.W + 5 * self.R + 3


			self.nn_size = nn_size
			self.interface_weigts = tf.Variable(tf.truncated_normal([self.batch_size,self.nn_size, self.interface_size], stddev=0.1))
			self.output_size =  output_size
			self.output_weights = tf.Variable(tf.truncated_normal([self.batch_size, self.W,self.output_size],stddev=0.1))
			self.interface_bias = tf.Variable(tf.fill([self.batch_size,1,output_size],0.1))
			self.interface_vector = tf.truncated_normal([self.batch_size, self.interface_size], stddev=0.1)


			# maps the indecies from the 2D array of free list per batch to
			# their corresponding values in the flat 1D array of ordered_allocation_weighting
			self.index_mapper = tf.constant(
			    np.cumsum([0] + [self.N] * (self.batch_size - 1), dtype=np.int32)[:, np.newaxis]
				)

			# def D(self,u,v):
			# 	"""
			# 	cosine similarity
			# 	"""
			# 	return tf.reduce_sum(tf.multiply(u,v)/tf.sqrt(tf.multiply(u,u) * tf.multiply(v,v)))


	def make_request(self,vec):
		with tf.variable_scope('memory'):
			self.interface_vector = tf.matmul(vec,self.interface_weigts)

			print "interface vector shape ", self.interface_vector.get_shape()

			self.interface_vector = tf.squeeze(self.interface_vector,[1])

			#***********************************************************
			parsed = {}

			r_keys_end = self.W * self.R
			r_strengths_end = r_keys_end + self.R
			w_key_end = r_strengths_end + self.W
			erase_end = w_key_end + 1 + self.W
			write_end = erase_end + self.W
			free_end = write_end + self.R

			r_keys_shape = (-1, self.W, self.R)
			r_strengths_shape = (-1, self.R)
			w_key_shape = (-1, self.W, 1)
			write_shape = erase_shape = (-1, self.W)
			free_shape = (-1, self.R)
			modes_shape = (-1, 3, self.R)

			# parsing the vector into its individual components
			parsed['read_keys'] = tf.reshape(self.interface_vector[:, :r_keys_end], r_keys_shape)
			parsed['read_strengths'] = tf.reshape(self.interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape)
			parsed['write_key'] = tf.reshape(self.interface_vector[:, r_strengths_end:w_key_end], w_key_shape)
			parsed['write_strength'] = tf.reshape(self.interface_vector[:, w_key_end], (-1, 1))
			parsed['erase_vector'] = tf.reshape(self.interface_vector[:, w_key_end + 1:erase_end], erase_shape)
			parsed['write_vector'] = tf.reshape(self.interface_vector[:, erase_end:write_end], write_shape)
			parsed['free_gates'] = tf.reshape(self.interface_vector[:, write_end:free_end], free_shape)
			parsed['allocation_gate'] = tf.expand_dims(self.interface_vector[:, free_end], 1)
			parsed['write_gate'] = tf.expand_dims(self.interface_vector[:, free_end + 1], 1)
			parsed['read_modes'] = tf.reshape(self.interface_vector[:, free_end + 2:], modes_shape)

			# transforming the components to ensure they're in the right ranges
			parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
			parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
			parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
			parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
			parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
			parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
			parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)









			#***********************************************************




			# partition = tf.constant([[0]*(self.R*self.W) + [1]*(self.R) + [2]*(self.W) + [3] + \
	  #                   [4]*(self.W) + [5]*(self.W) + \
	  #                   [6]*(self.R) + [7] + [8] + [9]*(self.R*3)], dtype=tf.int32)

			# (read_keys, read_strengths, write_key, write_strength,
	  #        erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes) = tf.dynamic_partition(self.interface_vec, partition, 10)
	        

			read_keys = parsed['read_keys']#tf.reshape(read_keys,[self.R, self.W]) #R*W
			self.read_keys = read_keys
			read_strengths = 1 + tf.nn.softplus(parsed['read_strengths'])#1 + tf.nn.softplus(tf.reshape(read_strengths, [1,self.R])) #1*R
			self.read_strengths = read_strengths
	        
	        #write vectors
			write_key = parsed['write_key']  #tf.reshape(write_key, [1,self.W]) #1*W
			self.write_key = write_key
	        #help init our write weights
			write_strength = 1 + tf.nn.softplus(parsed['write_strength']) #1 + tf.nn.softplus(tf.reshape(write_strength, [1,1])) #1*1
			self.write_strength = write_strength
			erase_vec = tf.nn.sigmoid(parsed['erase_vector'])#tf.nn.sigmoid(tf.reshape(erase_vec, [1,self.W])) #1*W
			self.erase_vec = erase_vec
			write_vec = parsed['write_vector'] #tf.reshape(write_vec, [1,self.W]) #1*W
			self.write_vec = write_vec
	        
	        #the degree to which locations at read heads will be freed
			free_gates = tf.nn.sigmoid(parsed['free_gates']) #tf.nn.sigmoid(tf.reshape(free_gates, [1,self.R])) #1*R
			self.free_gates = free_gates
	        #the fraction of writing that is being allocated in a new location
			allocation_gate = tf.nn.sigmoid(parsed['allocation_gate']) #tf.nn.sigmoid(allocation_gate) #1
			self.allocation_gate = allocation_gate
	        #the amount of information to be written to memory
			write_gate = tf.nn.sigmoid(parsed['write_gate']) #tf.nn.sigmoid(write_gate) #1
			self.write_gate = write_gate
	        #the softmax distribution between the three read modes (backward, forward, lookup)
	        #The read heads can use gates called read modes to switch between content lookup 
	        #using a read key and reading out locations either forwards or backwards 
	        #in the order they were written.
			read_modes = tf.nn.softmax(parsed['read_modes'], 1) #tf.nn.softmax(tf.reshape(read_modes, [3, self.R])) #3*R
			self.read_modes = read_modes
	        

			self.read_vecs = self.read_and_write(read_keys, read_strengths, write_key, write_strength, erase_vec, write_vec, free_gates, allocation_gate, write_gate, read_modes)
			self.read_vecs = tf.matmul(self.read_vecs,  self.output_weights)
			print('read_vecs ', self.read_vecs.get_shape())
			print('interface bias ', self.interface_bias.get_shape())

			return self.read_vecs + self.interface_bias


	def content_based_addressing(self,k, beta):
		"""
		k -- lookup key 
		beta -- scalar [1,inf), strength
		"""
		
		normed_k = tf.nn.l2_normalize(k,1)
		#normed_k = tf.reshape(normed_k,[self.W,1])
		normed_memory = tf.nn.l2_normalize(self.M,2)
		beta = tf.expand_dims(beta, 1)

		return tf.nn.softmax(tf.matmul(normed_memory,normed_k) * beta,0)

		# summ = 0
		# all_contents = []
		# for i in range(0,self.N):
		# 	contents = tf.exp(self.D(k,self.M[i])*beta)
		# 	#print contents.get_shape()
		# 	all_contents.append(contents)
		# 	summ += contents

		# content = tf.Variable(all_contents)

		# content = content * (1/summ)	
		

		# return content

	def retention_vec_update(self,f):
		# mul_res = 1

		#print f.get_shape()

		#Only for R=1


		self.psi = tf.reduce_prod(1 - tf.expand_dims(f,1) * self.read_weighting,2)


	def usage_vec_update(self,free_gates):
		self.retention_vec_update(free_gates)
		self.u = (self.u + self.write_weighting - self.u * self.write_weighting) * self.psi

	def get_allocation_vec(self, free_gates):
		#Maybe not correct
		#????????????????????????

		self.usage_vec_update(free_gates)
		sorted_usage, free_list = tf.nn.top_k(-1 * self.u, self.N)
		sorted_usage = -1 * sorted_usage

		shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
		unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

		mapped_free_list = free_list + self.index_mapper
		flat_unordered_allocation_weighting = tf.reshape(unordered_allocation_weighting, (-1,))
		flat_mapped_free_list = tf.reshape(mapped_free_list, (-1,))
		flat_container = tf.TensorArray(tf.float32, self.batch_size * self.N)

		flat_ordered_weightings = flat_container.scatter(
			flat_mapped_free_list,
			flat_unordered_allocation_weighting
		)

		packed_wightings = flat_ordered_weightings.stack()
		return tf.reshape(packed_wightings, (self.batch_size, self.N))




		# self.usage_vec_update(free_gates)
		# sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.u, self.N)
		# sorted_usage_vec *= -1
		# sorted_usage_cumprod = tf.cumprod(sorted_usage_vec,axis=0,exclusive=True)
		# unorder = (1-sorted_usage_vec)*sorted_usage_cumprod

		# alloc_weights = tf.zeros([self.N])
		# I = tf.constant(np.identity(self.N, dtype=np.float32))
        
#       #for each usage vec
		# for pos, idx in enumerate(tf.unstack(free_list[0])):
#           #flatten
		# 	m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
#           #add to weight matrix
		# 	alloc_weights += m*unorder[0, pos]
#       #the allocation weighting for each row in memory
		# return tf.reshape(alloc_weights, [1, self.N])

		# return (1 - sorted_usage_vec) * sorted_usage_cumprod

	def write_weighting_update(self,free_gates,k_write,beta_write,g_write,g_allocation):

		c = self.content_based_addressing(k_write,beta_write) # t-1
		a = self.get_allocation_vec(free_gates)
		# self.last_write_weighting = self.write_weighting
		
		c = tf.squeeze(c)

		print 'g_write ', g_write.get_shape()
		print 'g_allocation ', g_allocation.get_shape()
		print 'a ', a.get_shape()
		print 'c ', c.get_shape()



		print "write weighting before ", self.write_weighting.get_shape()
		self.write_weighting = g_write * (g_allocation * a + (1 - g_allocation) * c)
		print "write weighting after ", self.write_weighting.get_shape()


	def pairwise_add(self, u, v=None, is_batch=False):
	    """
	    performs a pairwise summation between vectors (possibly the same)
	    Parameters:
	    ----------
	    u: Tensor (n, ) | (n, 1)
	    v: Tensor (n, ) | (n, 1) [optional]
	    is_batch: bool
	        a flag for whether the vectors come in a batch
	        ie.: whether the vectors has a shape of (b,n) or (b,n,1)
	    Returns: Tensor (n, n)
	    Raises: ValueError
	    """
	    u_shape = u.get_shape().as_list()

	    if len(u_shape) > 2 and not is_batch:
	        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
	    if len(u_shape) > 3 and is_batch:
	        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

	    if v is None:
	        v = u
	    else:
	        v_shape = v.get_shape().as_list()
	        if u_shape != v_shape:
	            raise VauleError("Shapes %s and %s do not match" % (u_shape, v_shape))

	    n = u_shape[0] if not is_batch else u_shape[1]

	    print "u_shape ", u_shape
	    print "n ", n

	    column_u = tf.reshape(u, (-1, 1) if not is_batch else (-1, n, 1))
	    U = tf.concat([column_u] * n, 1 if not is_batch else 2)

	    if v is u:
	        return U + tf.transpose(U, None if not is_batch else [0, 2, 1])
	    else:
	        row_v = tf.reshape(v, (1, -1) if not is_batch else (-1, 1, n))
	        V = tf.concat([row_v] * n,0 if not is_batch else 1)

	        return U + V


	def update_temporal_memory_update(self):

		write_weighting = tf.expand_dims(self.write_weighting, 2)
		precedence_vector = tf.expand_dims(self.p, 1)

		print "write weighting temporal", write_weighting.get_shape()
		print "precedence temporal ", precedence_vector.get_shape()
		print "L before ", self.L.get_shape()

		reset_factor = 1 - self.pairwise_add(write_weighting, is_batch=True)
		self.L = reset_factor * self.L + tf.matmul(write_weighting, precedence_vector)
		self.L = (1 - tf.eye(self.N)) * self.L  # eliminates self-links

		print "L after ", self.L.get_shape()		

		# self.L = (1 - nnweight_vec - tf.transpose(nnweight_vec))*self.L + \
		                # tf.matmul(self.write_weighting, self.p, transpose_b=True)
		# print "a ", a.get_shape()
		# print "p ", self.p.get_shape()

		# b = tf.multiply(a ,self.p)
		# self.p = b + self.write_weighting

		# for i in range(0, self.N):
			# for j in range(0, self.N):
				# self.L[i][j] = (1 - self.write_weighting[i] - self.write_weighting[j]) * self.L[i][j]  + self.write_weighting[i] * self.p[j] 

		# self.L = (1.0 - tf.eye(self.N)) * self.L


		reset_factor_p = 1 - tf.reduce_sum(self.write_weighting, 1, keep_dims=True)
		self.p = reset_factor_p * self.p + self.write_weighting

		print "precedence after ", self.p.get_shape()

		# self.p = (1-tf.reduce_sum(self.write_weighting, reduction_indices=0)) * \
		                         # self.p + self.write_weighting

		# print "forward before ", self.forward.get_shape()
		self.forward = tf.matmul(self.read_weighting, self.L,) # t-1
		print "forward after ", self.forward.get_shape()

		# print "backward before ", self.backward.get_shape()
		self.backward = tf.matmul(self.read_weighting,tf.transpose(self.L,perm=[0, 2, 1])) #t-1
		print "backward after ", self.backward.get_shape()


	def update_read_weighting(self,k_read,beta_r, pi):
		c = self.content_based_addressing(k_read,beta_r)
		self.c = c
		# print "update read weighting before", self.read_weighting.get_shape()
		# self.last_read_weighting = self.read_weighting

		print "backward ", self.backward.get_shape()
		print "forward ", self.forward.get_shape()
		print "content ", c.get_shape()



		self.read_weighting = tf.expand_dims(pi[:, 0, :], 1)*self.backward + \
							tf.expand_dims(pi[:, 1, :], 1)*tf.transpose(c, perm=[0, 2, 1]) \
							+ tf.expand_dims(pi[:, 2, :], 1)*self.forward

		print "update read weighting after", self.read_weighting.get_shape()

	def get_read_vec(self):
		print "read weighting ", self.read_weighting.get_shape()
		print "memory ", self.M.get_shape()
		return tf.matmul(self.read_weighting,self.M)

	def update_memory(self,e,v):

		print 'write_weighting ', self.write_weighting.get_shape()
		
		self.M_before = self.M
		self.M = self.M*(1-tf.matmul(tf.transpose(self.write_weighting), e)) + \
                       tf.matmul(tf.transpose(self.write_weighting), v)

		# a = 1 - tf.matmul(tf.transpose(self.write_weighting), e)

		# self.M = self.M * a + tf.matmul(tf.transpose(self.write_weighting), v)


	def read_and_write(self,read_keys, read_strengths, key_write, write_strength, erase_vec, 
			write_vec, free_gates, allocation_gate, write_gate, read_modes):

		print "#######"
		print 'read_keys ', read_keys.get_shape()
		print 'read_strengths ', read_strengths.get_shape()
		print 'key_write ', key_write.get_shape()
		print 'write_strength ', write_strength.get_shape()
		print 'erase_vec ', erase_vec.get_shape()
		print 'write_vec ', write_vec.get_shape()
		print 'free_gates ', free_gates.get_shape()
		print 'allocation_gate ', allocation_gate.get_shape()
		print 'write_gate ', write_gate.get_shape()
		print 'read_modes ', read_modes.get_shape()
		print "########"



		self.write_weighting_update(free_gates,key_write,write_strength,write_gate,allocation_gate)
		self.update_memory(erase_vec,write_vec)	
		self.update_temporal_memory_update()


		self.update_temporal_memory_update()
		self.update_read_weighting(read_keys, read_strengths, read_modes)

		return self.get_read_vec()


		