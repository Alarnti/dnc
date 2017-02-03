import numpy as np
from numpy import linalg as la
import tensorflow as tf


class Memory:

	self.R = 1
	self.N = 5
	self.W = 10
	self.M = tf.Variable(tf.zeros(self.N,self.W))

	self.psi = tf.Variable(tf.zeros(1,self.N))
	self.u = tf.Variable(tf.zeros(tf.zeros(1,self.N)))

	self.weight = tf.Variable(tf.fill((self.R,self.W),1e-3))

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
			mul_res *= 1 - f[i] * self.weight[i]

		self.psi = mul_res

	def usage_vec_update():
		self.u = tf.mul(self.u + self.weight - tf.mul(self.u,self.weight),self.psi)

	def allocation_vec(self,f):
		#retention vector
		ret_vec = np.ones((1,self.N))

		for t in range(0, self.N):
			for i in range(0,self.R):
				ret_vec[t] *= 1 - np.multiply(f[i],M[t])

		#usage vector
		
		u = (u + )

