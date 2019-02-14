import tensorflow as tf
import numpy as np
import cv2 as cv
import encoder
import os

class recog:
	def __init__(self, threshold_1 = 3e-06, threshold_2 = 6.0e-5, path = './face_recognition/saved_model/beta', name = 'alpha'):
		self.threshold1 = tf.constant(threshold_1)
		self.threshold2 = tf.constant(threshold_2)
		self.input = tf.placeholder(tf.int32)
		self.encoder = encoder.encoder(self.input, name)
		self.centroid = None
		self.path = path

	def get_result(self, batch, session):
		if self.centroid is None:
			images = np.zeros(shape = [0, 64, 64, 3])
			for file in os.listdir('face_recognition/my_train_images'):
				image = cv.resize(cv.imread('face_recognition/my_train_images/' + file), (64, 64), interpolation = cv.INTER_CUBIC)
				image = np.asarray(image).reshape([1, 64, 64, 3])
				images = np.concatenate([images, image])
			self.centroid = tf.constant(np.asarray(session.run(self.encoder.embedding, feed_dict = {self.input: images})).reshape([1, -1, 64]))
			self.difference = tf.reduce_sum(tf.square(self.centroid - tf.expand_dims(self.encoder.embedding, 1)), axis = 2)
			self.min_diff = tf.reduce_min(self.difference, axis = 1)
			self.sum_diff = tf.reduce_sum(self.difference, axis = 1)
			self.output = tf.math.logical_and((self.min_diff <= self.threshold1), (self.sum_diff <= self.threshold2))
			print('Centroid Processed')
		print(session.run(self.min_diff, feed_dict = {self.input: batch}), end = ', ')
		print(session.run(self.sum_diff, feed_dict = {self.input: batch}))
		return session.run(self.output, feed_dict = {self.input: batch})

