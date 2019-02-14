import tensorflow as tf
import numpy as np
import cv2 as cv
import segnet
THRESHOLD = 0.5
class seg:
	def __init__(self, path = './face_detection/saved_model/model'):
		import tensorflow as tf
		self.net = segnet.segnet()
		self.output = self.net.output
		self.path = path
	
	def segment(self, batch_inputs, session):
		output = session.run(255 * tf.cast((self.net.segmented_output >= THRESHOLD), dtype = tf.int32), feed_dict = {self.net.input_layer: batch_inputs})
		return output.reshape([200, 200, 1])
