import encoder
import densenet
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import random


LEARNING_RATE = 0.00001
EPOCHS = 200
BATCH_SIZE = 16


class trainer:
	def __init__(self, trained_model_path = './saved_model/alpha', name = 'alpha'):
		self.input = tf.placeholder(tf.int32)
		self.encoder = encoder.encoder(self.input, name)
		self.learning_rate = tf.placeholder(tf.float32)
		self.labels = tf.placeholder(tf.int32)
		self.densenet = densenet.densenet(self.encoder, name)
		self.trainable_vars = self.densenet.get_var_list()
		self.output = self.densenet.output
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.densenet.denseoutput4, labels = tf.cast(self.labels, tf.float32) / 255.0))
		self.session = tf.Session()
		self.output_verdict = tf.cast((self.output >= 0.5), tf.int32)
		self.correct = tf.equal(self.output_verdict, self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct, dtype = tf.float32))
		self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list = [self.trainable_vars])
		if trained_model_path is None:
			self.session.run(tf.global_variables_initializer())
		else:
			self.session.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			saver.restore(self.session, trained_model_path)
	def train_model(self, batch, labels, learning_rate = 0.001):
		self.session.run(self.train, feed_dict = {self.input: batch, self.densenet.dropout: 0.5, self.labels: labels, self.learning_rate: 0.001})

	def get_result(self, batch):
		return self.session.run(self.output, feed_dict = {self.input: batch, self.densenet.dropout: 1.0})

	def get_loss(self, batch, labels):
		return self.session.run(self.loss, feed_dict = {self.input: batch, self.densenet.dropout: 1.0, self.labels: labels})

	def save_model(self, path = 'saved_model/beta'):
		saver = tf.train.Saver()
		saver.save(self.session, path)
	def get_accuracy(self, batch, labels):
		return 100.0 * self.session.run(self.accuracy, feed_dict = {self.input: batch, self.densenet.dropout: 1.0, self.labels: labels})

###########		load data
#load positive examples
print('Loading Data...')
positive_images = np.zeros(shape = [0, 64, 64, 3])
negative_images = np.zeros(shape = [0, 64, 64, 3])
for file in os.listdir('my_train_images'):
	image = np.asarray(cv.imread('my_train_images/' + file)).reshape([1, 64, 64, 3])
	positive_images = np.concatenate([positive_images, image])
negatives_lo_load = random.sample(range(0, 13233), np.shape(positive_images)[0] * 2)
for i in negatives_lo_load:
	filename = 'lfwcrop_color/faces/' + str(i) + '.ppm'
	image = np.asarray(cv.imread(filename)).reshape([1, 64, 64, 3])
	negative_images = np.concatenate([negative_images, image])

positive_labels = np.ones(shape = [np.shape(positive_images)[0]], dtype = int)
negative_labels = np.zeros(shape = [np.shape(negative_images)[0]], dtype = int)

images = np.concatenate([positive_images, negative_images])
labels = np.concatenate([positive_labels, negative_labels])

test_cnt = 10 * len(images) // 100
positive_test_cnt = test_cnt // 3
negative_test_cnt = test_cnt - positive_test_cnt
test_images = np.concatenate([images[: positive_test_cnt], images[-negative_test_cnt: ]])
test_labels = np.concatenate([labels[: positive_test_cnt], labels[-negative_test_cnt: ]])
images = images[positive_test_cnt: -negative_test_cnt]
labels = labels[positive_test_cnt: -negative_test_cnt]

random_indices = np.random.permutation(len(images))
images, labels = (images[random_indices], labels[random_indices])

print('Data Loaded')
#############Data Loaded

trainer_obj = trainer(trained_model_path = None)
max_acc = 0
for epoch in range(EPOCHS):
	for batch_no in range((np.shape(images)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		L = BATCH_SIZE * batch_no
		R = L + BATCH_SIZE
		batch_x = images[L: R]
		batch_y = labels[L: R]
		trainer_obj.train_model(batch_x, batch_y, LEARNING_RATE)
	training_loss, test_loss = (0.0, 0.0)
	for batch_no in range((np.shape(images)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		L = BATCH_SIZE * batch_no
		R = L + BATCH_SIZE
		batch_x = images[L: R]
		batch_y = labels[L: R]
		training_loss += trainer_obj.get_loss(batch_x, batch_y)
	for batch_no in range((np.shape(test_images)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		L = BATCH_SIZE * batch_no
		R = L + BATCH_SIZE
		batch_x = test_images[L: R]
		batch_y = test_labels[L: R]
		test_loss += trainer_obj.get_loss(batch_x, batch_y)
	test_accuracy = trainer_obj.get_accuracy(test_images, test_labels)
	print('Epoch: {:10d}\tTrain-loss: {:10f}\tTest-loss: {:10f}\tTest-accuracy: {:10f}'.format(epoch, training_loss, test_loss, test_accuracy))
	if(test_accuracy > max_acc):
		trainer_obj.save_model()
		print('Saving Model...')
		max_acc = test_accuracy
