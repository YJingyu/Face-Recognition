class segnet:
	def __init__(self):
		import tensorflow as tf
		self.input_layer = tf.placeholder(dtype = tf.int32)
		self.normalized_input_layer = tf.cast(self.input_layer, dtype = tf.float32) / 255.0
		###########################
		self.wconv1 = tf.get_variable(name = 'wconv1', dtype = tf.float32, shape = [7, 7, 3, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv1 = tf.get_variable(name = 'bconv1', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv1 = tf.nn.conv2d(self.normalized_input_layer, self.wconv1, padding = 'VALID', strides = [1, 1, 1, 1]) + self.bconv1
		self.conv1 = tf.nn.relu(self.conv1)
		#now the shape is 194 x 194 x 32
		##########################
		self.wconv2 = tf.get_variable(name = 'wconv2', dtype = tf.float32, shape = [7, 7, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv2 = tf.get_variable(name = 'bconv2', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv2 = tf.nn.conv2d(self.conv1, self.wconv2, padding = 'VALID', strides = [1, 2, 2, 1]) + self.bconv2
		self.conv2 = tf.nn.relu(self.conv2)
		#now the shape is 94 x 94 x 32
		##########################
		self.wconv3 = tf.get_variable(name = 'wconv3', dtype = tf.float32, shape = [5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv3 = tf.get_variable(name = 'bconv3', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(0.0))
		self.conv3 = tf.nn.conv2d(self.conv2, self.wconv3, padding = 'VALID', strides = [1, 2, 2, 1]) + self.bconv3
		self.conv3 = tf.nn.relu(self.conv3)
		#now the output is 45 x 45 32
		##########################
		self.wconv4 = tf.get_variable(name = 'wconv4', dtype = tf.float32, shape = [5, 5, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv4 = tf.get_variable(name = 'bconv4', dtype = tf.float32, shape = [64], initializer = tf.initializers.constant(1.0))
		self.conv4 = tf.nn.conv2d(self.conv3, self.wconv4, padding = 'VALID', strides = [1, 4, 4, 1]) + self.bconv4
		self.conv4 = tf.nn.relu(self.conv4)
		#now the output shape is 11 x 11 x 64
		##########################
		#Encoder Portion done
		##########################
		self.upsample4 = tf.image.resize_bilinear(self.conv4, size = [45, 45])
		self.wconv5 = tf.get_variable(name = 'wconv5', dtype = tf.float32, shape = [5, 5, 64, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv5 = tf.get_variable(name = 'bconv5', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv5 = tf.nn.conv2d(self.upsample4, self.wconv5, strides = [1, 1, 1, 1], padding ='SAME') + self.bconv5
		self.conv5 = tf.nn.relu(self.conv5)
		#now the output shape is 45 x 45 x 32
		###########################
		self.upsample5 = tf.image.resize_bilinear(self.conv5, size = [94, 94])
		self.wconv6 = tf.get_variable(name = 'wconv6', dtype = tf.float32, shape = [5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv6 = tf.get_variable(name = 'bconv6', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(0.0))
		self.conv6 = tf.nn.conv2d(self.upsample5, self.wconv6, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv6
		self.conv6 = tf.nn.relu(self.conv6)
		#output shape is 94 x 94 x 32
		##########################
		self.upsample6 = tf.image.resize_bilinear(self.conv6, size = [194, 194])
		self.wconv7 = tf.get_variable(name = 'wconv7', dtype = tf.float32, shape = [7, 7, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv7 = tf.get_variable(name = 'bconv7', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(1.0))
		self.conv7 = tf.nn.conv2d(self.upsample6, self.wconv7, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv7
		self.conv7 = tf.nn.relu(self.conv7)
		#output is of shape 194 x 194 x 32
		###########################
		self.upsample7 = tf.image.resize_bilinear(self.conv7, size = [200, 200])
		self.wconv8 = tf.get_variable(name = 'wconv8', dtype = tf.float32, shape = [7, 7, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv8 = tf.get_variable(name = 'bconv8', dtype = tf.float32, shape = [32], initializer = tf.initializers.constant(0.0))
		self.conv8 = tf.nn.conv2d(self.upsample7, self.wconv8, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv8
		self.conv8 = tf.nn.relu(self.conv8)
		###########################
		#final convolutional layer
		self.wconv9 = tf.get_variable(name = 'wconv9', dtype = tf.float32, shape = [7, 7, 32, 1], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv9 = tf.get_variable(name = 'bconv9', dtype = tf.float32, shape = [1], initializer = tf.initializers.constant(0.0))
		self.conv9 = tf.nn.conv2d(self.conv8, self.wconv9, padding = 'SAME', strides = [1, 1, 1, 1]) + self.bconv9

		self.output = self.conv9
		self.segmented_output = tf.nn.sigmoid(self.output)
