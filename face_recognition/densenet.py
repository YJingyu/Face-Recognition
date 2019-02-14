import tensorflow as tf

class densenet:
	def __init__(self, encoder, name = ''):
		# 64 32 32 1
		self.dropout = tf.placeholder(tf.float32)
		self.w1 = tf.get_variable(name = name + 'w1', dtype = tf.float32, shape = [64, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.b1 = tf.get_variable(name = name + 'b1', dtype = tf.float32, shape = [64], initializer = tf.initializers.constant(1.0))
		self.denseoutput1 = tf.matmul(encoder.embedding, self.w1) + self.b1
		self.denseoutput1 = tf.nn.relu(self.denseoutput1)

		self.w2 = tf.get_variable(name = name + 'w2', dtype = tf.float32, shape = [64, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.b2 = tf.get_variable(name = name + 'b2', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.denseoutput2 = tf.matmul(self.denseoutput1, self.w2) + self.b2
		self.denseoutput2 = tf.nn.relu(self.denseoutput2)
		self.denseoutput2 = tf.nn.dropout(self.denseoutput2, keep_prob = self.dropout)

		self.w3 = tf.get_variable(name = name + 'w3', dtype = tf.float32, shape = [32, 16], initializer = tf.contrib.layers.xavier_initializer())
		self.b3 = tf.get_variable(name = name + 'b3', dtype = tf.float32, shape = [16], initializer = tf.initializers.constant(0.0))
		self.denseoutput3 = tf.matmul(self.denseoutput2, self.w3) + self.b3
		self.denseoutput3 = tf.nn.relu(self.denseoutput3)

		self.w4 = tf.get_variable(name = name + 'w4', dtype = tf.float32, shape = [16, 1], initializer = tf.contrib.layers.xavier_initializer())
		self.b4 = tf.get_variable(name = name + 'b4', dtype = tf.float32, shape = [1], initializer = tf.initializers.constant(0.0))
		self.denseoutput4 = tf.matmul(self.denseoutput3, self.w4) + self.b4
		self.output = tf.sigmoid(self.denseoutput4)

	def get_var_list(self):
		return [self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4]