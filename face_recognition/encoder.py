import tensorflow as tf
class encoder():
	def __init__(self, input_layer, name = ''):
		name = name + '-encoder-'
		self.normalized_input_layer = tf.cast(input_layer, dtype = tf.float32) / 255.0
		#64 x 64 x 3
		###########################
		self.wconv1 = tf.get_variable(name = name + 'wconv1', dtype = tf.float32, shape = [7, 7, 3, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv1 = tf.get_variable(name = name + 'bconv1', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv1 = tf.nn.conv2d(self.normalized_input_layer, self.wconv1, padding = 'VALID', strides = [1, 1, 1, 1]) + self.bconv1
		self.conv1 = tf.nn.lrn(tf.nn.relu(self.conv1))
		#now the shape is 58 x 58 x 32
		##########################
		self.wconv2 = tf.get_variable(name = name + 'wconv2', dtype = tf.float32, shape = [7, 7, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv2 = tf.get_variable(name = name + 'bconv2', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv2 = tf.nn.conv2d(self.conv1, self.wconv2, padding = 'VALID', strides = [1, 1, 1, 1]) + self.bconv2
		self.conv2 = tf.nn.lrn(tf.nn.relu(self.conv2))
		#now the shape is 52 x 52 x 32
		##########################
		self.wconv3 = tf.get_variable(name = name + 'wconv3', dtype = tf.float32, shape = [5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv3 = tf.get_variable(name = name + 'bconv3', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(0.0))
		self.conv3 = tf.nn.conv2d(self.conv2, self.wconv3, padding = 'VALID', strides = [1, 2, 2, 1]) + self.bconv3
		self.conv3 = tf.nn.lrn(tf.nn.relu(self.conv3))
		#now the output is 24 x 24 x 32
		##########################
		self.wconv4 = tf.get_variable(name = name + 'wconv4', dtype = tf.float32, shape = [5, 5, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv4 = tf.get_variable(name = name + 'bconv4', dtype = tf.float32, shape = [64], initializer = tf.initializers.constant(1.0))
		self.conv4 = tf.nn.conv2d(self.conv3, self.wconv4, padding = 'VALID', strides = [1, 2, 2, 1]) + self.bconv4
		self.conv4 = tf.nn.lrn(tf.nn.relu(self.conv4))
		#now the output shape is 10 x 10 x 64
		##########################
		self.wconv5 = tf.get_variable(name = name + 'wconv5', dtype = tf.float32, shape = [5, 5, 64, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv5 = tf.get_variable(name = name + 'bconv5', dtype = tf.float32, shape = [64], initializer = tf.contrib.layers.xavier_initializer())
		self.conv5 = tf.nn.conv2d(self.conv4, self.wconv5, padding = 'VALID', strides = [1, 1, 1, 1]) + self.bconv5
		self.conv5 = tf.nn.lrn(tf.nn.relu(self.conv5))
		#now the shape is 6 x 6 x 64
		##########################
		#Now the fully connected encoder portion
		self.wfc_encoder = tf.get_variable(name = name + 'wfc_encoder', dtype = tf.float32, shape = [6 * 6 * 64, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bfc_encoder = tf.get_variable(name = name + 'bfc_encoder', dtype = tf.float32, shape = [64], initializer = tf.initializers.constant(1.0))
		self.fc_encoder = tf.matmul(tf.reshape(self.conv5, shape = [-1, 6 * 6 * 64]), self.wfc_encoder) + self.bfc_encoder
		##########################
		#Latent space:
		self.embedding = self.fc_encoder
