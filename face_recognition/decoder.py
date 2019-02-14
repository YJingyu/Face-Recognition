import tensorflow as tf
class decoder():
	def __init__(self, embedding, name = ''):
		name = name + '-decoder-'
		self.wfc_decoder = tf.get_variable(name = name + 'wfc_decoder', dtype = tf.float32, shape = [64, 6 * 6 * 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bfc_decoder = tf.get_variable(name = name + 'bfc_decoder', dtype = tf.float32, shape = [6 * 6 * 64], initializer = tf.initializers.constant(1.0))
		self.fc_decoder = tf.matmul(embedding, self.wfc_decoder) + self.bfc_decoder
		##########################
		self.upsample5 = tf.image.resize_bilinear(tf.reshape(self.fc_decoder, shape = [-1, 6, 6, 64]), size = [10, 10])
		self.wconv6 = tf.get_variable(name = name + 'wconv6', dtype = tf.float32, shape = [5, 5, 64, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv6 = tf.get_variable(name = name + 'bconv6', dtype = tf.float32, shape = [64], initializer = tf.initializers.constant(1.0))
		self.conv6 = tf.nn.conv2d(self.upsample5, self.wconv6, strides = [1, 1, 1, 1], padding ='SAME') + self.bconv6
		self.conv6 = tf.nn.lrn(tf.nn.relu(self.conv6))
		#now the output shape is 10 x 10 x 64
		###########################
		self.upsample6 = tf.image.resize_bilinear(self.conv6, size = [24, 24])
		self.wconv7 = tf.get_variable(name = name + 'wconv7', dtype = tf.float32, shape = [5, 5, 64, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv7 = tf.get_variable(name = name + 'bconv7', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(0.0))
		self.conv7 = tf.nn.conv2d(self.upsample6, self.wconv7, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv7
		self.conv7 = tf.nn.lrn(tf.nn.relu(self.conv7))
		#output shape is 24 x 24 x 32
		##########################
		self.upsample7 = tf.image.resize_bilinear(self.conv7, size = [52, 52])
		self.wconv8 = tf.get_variable(name = name + 'wconv8', dtype = tf.float32, shape = [5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv8 = tf.get_variable(name = name + 'bconv8', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv8 = tf.nn.conv2d(self.upsample7, self.wconv8, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv8
		self.conv8 = tf.nn.lrn(tf.nn.relu(self.conv8))
		#output is of shape 52 x 52 x 32
		###########################
		self.upsample8 = tf.image.resize_bilinear(self.conv8, size = [58, 58])
		self.wconv9 = tf.get_variable(name = name + 'wconv9', dtype = tf.float32, shape = [7, 7, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv9 = tf.get_variable(name = name + 'bconv9', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv9 = tf.nn.conv2d(self.upsample8, self.wconv9, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv9
		self.conv9 = tf.nn.lrn(tf.nn.relu(self.conv9))
		#output is of shape 58 x 58 x 32
		###########################
		#final convolutional layer
		self.upsample9 = tf.image.resize_bilinear(self.conv9, size = [64, 64])
		self.wconv10 = tf.get_variable(name = name + 'wconv10', dtype = tf.float32, shape = [7, 7, 32, 3], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv10 = tf.get_variable(name = name + 'bconv10', dtype = tf.float32, shape = [3], initializer = tf.initializers.constant(1.0))
		self.conv10 = tf.nn.conv2d(self.upsample9, self.wconv10, padding = 'SAME', strides = [1, 1, 1, 1]) + self.bconv10

		self.output = tf.nn.sigmoid(self.conv10)
		