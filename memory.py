import numpy as np
from numpy import linalg as la
import tensorflow as tf


class Memory:

	self.R = 1
	self.N = 5
	self.W = 10
	self.M = tf.Variable(tf.zeros(self.N,self.W))
	self.L = tf.Variable(tf.zeros(self.N,self.N))

	self.psi = tf.Variable(tf.zeros(1,self.N))
	self.u = tf.Variable(tf.zeros(tf.zeros(1,self.N)))
	self.free_list = tf.Variable(tf.zeros(1,self.N))
	self.write_weighting = tf.Variable(tf.fill((self.R,self.W),1e-3))
	self.read_weighting = 
	self.p = tf.Variable(tf.zeros(1,self.N))
	self.forward = 0
	self.backward = 0

	self.size_ksi = self.W*self.R + 3 * self.W + 5 * self.R + 3


	def D(u,v):
		"""
		cosine similarity
		"""

		return tf.mul(u,v)/sqrt(tf.mul(u,u) * tf.mul(v,v))

	def content_based_addressing(self,k, beta):
		"""
		k -- lookup key []W
		beta -- scalar [1,inf), strength

		"""

		contents = tf.Variable(tf.zeros(1,self.N))
		for i in range(0,self.N):
			contents = tf.exp(D(k,M[i])*beta)
		summ = sum(contents)

		contents = tf.mul(contents,1/summ)
		

		return contents

	def retention_vec_update(f):
		mul_res = 1

		for i in range(0,self.R):
			mul_res *= 1 - f[i] * self.write_weighting[i]

		self.psi = mul_res

	def usage_vec_update():
		self.u = tf.mul(self.u + self.write_weighting - tf.mul(self.u,self.write_weighting),self.psi)

	def get_allocation_vec():
		sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.u, self.N)
		sorted_usage_cumprod = tf.cumprod(sorted_usage_vec)

		return (1 - sorted_usage_vec) * sorted_usage_cumprod

	def write_weight_update(k_write,beta_write,g_write,g_allocation):

		c = self.content_based_addressing(k_write,beta) # t-1
		a = get_allocation_vec()
		self.write_weighting = g_write * (g_allocation * a + (1 - g_allocation) * c)


	def update_temporal_memory_update():
		p = (1 - tf.sum(self.write_weighting)) * p + self.write_weighting

		self.L = 

		for i in range(0, self.N):
			for j in range(0, self.N):
				self.L[i][j] = (1 - self.write_weighting[i] - self.write_weighting[j]) * self.L[i][j]  + self.write_weighting[i] * self.p[j] 

		self.L = (1 - tf.eye(self.N)) * self.L

		self.forward = self.L * self.read_weighting # t-1
		self.backward = tf.transpose(self.L) * self.read_weighting #t-1


	def update_read_weighting(k_read,beta_r, pi):
		c = self.content_based_addressing(k_read,b_read)

		self.read_weighting = pi[0]*self.backward + pi[1]*c + pi[2]*self.forward

	def get_read_vec():
		return tf.transpose(self.M) * self.read_weighting

	def update_memory():
		self.M = tf.mul(self.M, tf.constant(np.identity(self.N))


