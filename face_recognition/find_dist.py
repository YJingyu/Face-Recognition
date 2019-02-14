import tensorflow as tf
import numpy as np
import cv2 as cv
import encoder
import densenet

class embed:
	def __init__(self, path = './face_recognition/saved_model/beta', name = 'alpha'):
		self.input = tf.placeholder(tf.int32)
		self.encoder = encoder.encoder(self.input, name)
		self.output = self.encoder.embedding
		self.path = path

	def get_embedding(self, batch, session):
		return session.run(self.output, feed_dict = {self.input: batch, self.densenet.dropout: 1.0})

a = embed()