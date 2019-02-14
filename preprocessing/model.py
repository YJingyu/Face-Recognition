import segnet
import tensorflow as tf
class model:
	def __init__(self, path = None):
		import tensorflow as tf
		self.net = segnet.segnet()
		self.learning_rate = tf.placeholder(dtype = tf.float32)
		self.labels = tf.placeholder(dtype = tf.int32)
		self.label_boolean = tf.reshape(tf.cast(self.labels / 255, dtype = tf.bool), [-1])
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.net.output, labels = tf.cast(self.labels / 255, dtype = tf.float32))
		self.trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		self.session = tf.Session()
		self.output_boolean = tf.reshape((self.net.output >= 0.5), [-1])
		#iou between normalized_labels and output_boolean
		self.intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(self.output_boolean, self.label_boolean), tf.float32))
		self.union = tf.reduce_sum(tf.cast(tf.math.logical_or(self.output_boolean, self.label_boolean), tf.float32))
		self.IOU_metric = self.intersection / (self.union + 0.000000001)
		if path is None:
			self.session.run(tf.global_variables_initializer())
		else:
			saver = tf.train.Saver()
			saver.restore(self.session, path)
	
	def train_model(self, batch_inputs, batch_labels, learning_rate = 0.001):
		self.session.run(self.trainer, feed_dict = {self.labels: batch_labels, self.net.input_layer: batch_inputs, self.learning_rate: learning_rate})

	def segment(self, batch_inputs):
		output = self.session.run(self.net.segmented_output, feed_dict = {self.net.input_layer: batch_inputs})
		return output

	def cost(self, batch_inputs, batch_labels):
		cost_value = self.session.run(self.loss, feed_dict = {self.net.input_layer: batch_inputs, self.labels: batch_labels})
		return cost_value

	def reset_weights(self):
		self.session.run(tf.global_variables_initializer())
	
	def get_IOU(self, batch_inputs, batch_labels):
		iou = self.session.run(self.IOU_metric, feed_dict = {self.net.input_layer: batch_inputs, self.labels: batch_labels})
		print (iou)
		return iou[0]