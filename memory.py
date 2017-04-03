import tensorflow as tf

class Memory:

	#only for R == 1

	def __init__(self):
		self.R = 1
		self.N = 5
		self.W = 10 #10
		self.M = tf.Variable(tf.zeros([self.N,self.W]))
		self.L = tf.Variable(tf.zeros([self.N,self.N]))

		self.psi = tf.Variable(tf.zeros([1,self.N]))
		self.u = tf.Variable(tf.zeros([1,self.N]))
		self.free_list = tf.Variable(tf.zeros(1,self.N))

		self.write_weighting = tf.Variable(tf.fill((1,self.N),1e-3))
		self.last_write_weighting = tf.Variable(tf.fill((1,self.N),1e-3))

		self.read_weighting = tf.Variable(tf.zeros([self.R,self.N]))
		self.last_read_weighting = tf.Variable(tf.zeros([self.R, self.N]))

		self.p = tf.Variable(tf.fill((1,self.N),0.0))
		self.forward = 0
		self.backward = 0

		self.size_ksi = self.W*self.R + 3 * self.W + 5 * self.R + 3

	def D(self,u,v):
		"""
		cosine similarity
		"""
		return tf.reduce_sum(tf.multiply(u,v)/tf.sqrt(tf.multiply(u,u) * tf.multiply(v,v)))

	def content_based_addressing(self,k, beta):
		"""
		k -- lookup key 
		beta -- scalar [1,inf), strength
		"""
		
		normed_k = tf.nn.l2_normalize(k,0)
		normed_k = tf.reshape(normed_k,[self.W,1])
		normed_memory = tf.nn.l2_normalize(self.M,1)

		return tf.nn.softmax(tf.matmul(normed_memory,normed_k) * beta)

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
		mul_res = 1

		#print f.get_shape()

		#Only for R=1


		self.psi = 1 - f * self.last_read_weighting


	def usage_vec_update(self,free_gates):
		self.retention_vec_update(free_gates)
		self.u = (self.u + self.last_write_weighting - self.u * self.last_write_weighting) * self.psi

	def get_allocation_vec(self, free_gates):
		#Maybe not correct

		self.usage_vec_update(free_gates)
		sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.u, self.N)
		sorted_usage_cumprod = tf.cumprod(sorted_usage_vec)

		return (1 - sorted_usage_vec) * sorted_usage_cumprod

	def write_weighting_update(self,free_gates,k_write,beta_write,g_write,g_allocation):

		c = self.content_based_addressing(k_write,beta_write) # t-1
		a = self.get_allocation_vec(free_gates)
		self.last_write_weighting = self.write_weighting
		
		print 'g_write ', g_write.get_shape()
		print 'g_allocation ', g_allocation.get_shape()
		print 'a ', a.get_shape()
		print 'c ', c.get_shape()


		print "write weighting before ", self.write_weighting.get_shape()
		self.write_weighting = g_write * (g_allocation * a + (1 - g_allocation) * tf.transpose(c))
		print "write weighting after ", self.write_weighting.get_shape()


	def update_temporal_memory_update(self):
		a = 1 - tf.reduce_sum(self.write_weighting)
		# print "a ", a.get_shape()
		# print "p ", self.p.get_shape()

		b = tf.multiply(a ,self.p)
		self.p = b + self.write_weighting

		# for i in range(0, self.N):
			# for j in range(0, self.N):
				# self.L[i][j] = (1 - self.write_weighting[i] - self.write_weighting[j]) * self.L[i][j]  + self.write_weighting[i] * self.p[j] 

		# self.L = (1.0 - tf.eye(self.N)) * self.L

		self.forward = tf.matmul(self.L, tf.transpose(self.read_weighting)) # t-1
		self.backward = tf.matmul(tf.transpose(self.L), tf.transpose(self.read_weighting)) #t-1


	def update_read_weighting(self,k_read,beta_r, pi):
		c = self.content_based_addressing(k_read,beta_r)

		print "updateread weighting before", self.read_weighting.get_shape()
		self.last_read_weighting = self.read_weighting

		print "backward ", self.backward.get_shape()
		print "forward ", self.forward.get_shape()
		print "content ", c.get_shape()

		self.read_weighting = pi[0]*self.backward + pi[1]*c + pi[2]*self.forward
		print "update read weighting after", self.read_weighting.get_shape()

	def get_read_vec(self):
		print "read weighting ", self.read_weighting.get_shape()
		print "memory ", self.M.get_shape()
		return tf.matmul(tf.transpose(self.M), self.read_weighting)

	def update_memory(self,e,v):

		print 'write_weighting ', self.write_weighting.get_shape()
		
		a = tf.ones([self.N,self.W]) - tf.matmul(tf.transpose(self.write_weighting), e)

		self.M = self.M * (a) + tf.matmul(tf.transpose(self.write_weighting), v)


	def read_and_write(self,read_keys, read_strengths, key_write, write_srength, erase_vec, 
			write_vec, free_gates, allocation_gate, write_gate, read_modes):

		print "#######"
		print 'read_keys ', read_keys.get_shape()
		print 'read_strengths ', read_strengths.get_shape()
		print 'key_write ', key_write.get_shape()
		print 'write_srength ', write_srength.get_shape()
		print 'erase_vec ', erase_vec.get_shape()
		print 'write_vec ', write_vec.get_shape()
		print 'free_gates ', free_gates.get_shape()
		print 'allocation_gate ', allocation_gate.get_shape()
		print 'write_gate ', write_gate.get_shape()
		print 'read_modes ', read_modes.get_shape()
		print "########"
		self.write_weighting_update(free_gates,key_write,write_srength,write_gate,allocation_gate)
		self.update_memory(erase_vec,write_vec)	
		self.update_temporal_memory_update()


		self.update_temporal_memory_update()
		self.update_read_weighting(read_keys, read_strengths, read_modes)

		return self.get_read_vec()


		