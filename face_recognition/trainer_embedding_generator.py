import tensorflow as tf
import encoder
import decoder
import numpy as np
import cv2 as cv

IMAGES = 13233
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32

class trainer:
	def __init__(self, saved_model_path = None, name = 'alpha'):
		self.input_layer = tf.placeholder(tf.float32)
		self.learning_rate = tf.placeholder(tf.float32)
		self.encoder_network = encoder.encoder(self.input_layer, name)
		self.decoder_network = decoder.decoder(self.encoder_network.embedding, name)
		###############	both have been linked
		self.loss = tf.reduce_mean(tf.square(self.decoder_network.output - self.encoder_network.normalized_input_layer))
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		self.session = tf.Session()
		if saved_model_path is None:
			self.session.run(tf.global_variables_initializer())
		else:
			saver = tf.train.Saver()
			saver.restore(self.session, saved_model_path)

	def train_network(self, batch, learning_rate):
		self.session.run(self.train, feed_dict = {self.input_layer: batch, self.learning_rate: learning_rate})

	def save_model(self, name = 'saved_model/alpha'):
		saver = tf.train.Saver()
		saver.save(self.session, name)

	def get_loss(self, batch):
		return self.session.run(self.loss, feed_dict = {self.input_layer: batch})


def load_data(L = 0, R = IMAGES):
	array = np.zeros([0, 64, 64, 3])
	for i in range(L, R):
		file_name = 'lfwcrop_color/faces/' + str(i) + '.ppm'
		image = np.asarray(cv.imread(file_name)).reshape([1, 64, 64, 3])
		flipped_image = np.asarray(cv.flip(image.reshape([64, 64, 3]), 1)).reshape([1, 64, 64, 3])
		image = np.concatenate((image, flipped_image))
		array = np.concatenate((array, image))
	return array

min_test_loss = 1e10
trainer_obj = trainer()
print('Loading Data: {} images to load'.format(IMAGES))
train_data = np.zeros(shape = [0, 64, 64, 3])
for i in range(0, IMAGES, 1024):
	L, R = (i, min(IMAGES, i + 1024))
	batch = load_data(L, R)
	print('Data Loaded from {} to {}'.format(L, R))
	train_data = np.concatenate((train_data, batch))
print('Data Loaded')
total = np.shape(train_data)[0]
print (total)
limit = total * 20 // 100
test_data = train_data[: limit]
train_data = train_data[limit: ]

for epoch in range(EPOCHS):
	test_loss, train_loss = (0.0, 0.0)
	test_batch_count, train_batch_count = (0, 0)
	for batch_no in range((np.shape(train_data)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch = train_data[L: R]
		trainer_obj.train_network(batch, LEARNING_RATE)

	for batch_no in range((np.shape(train_data)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		train_batch_count += 1
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch = train_data[L: R]
		train_loss += trainer_obj.get_loss(batch)

	for batch_no in range((np.shape(test_data)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		test_batch_count += 1
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch = test_data[L: R]
		test_loss += trainer_obj.get_loss(batch)
	test_loss /= test_batch_count
	train_loss /= train_batch_count
	print('Epoch: {:10d}\tTrain-loss: {:10f}\tTest-loss: {:10f}'.format(epoch, train_loss, test_loss))
	if(test_loss < min_test_loss):
		min_test_loss = test_loss
		print('Saving Model...')
		trainer_obj.save_model()
		if(epoch >= 50):
			LEARNING_RATE /= 2.0
	np.random.shuffle(test_data)
