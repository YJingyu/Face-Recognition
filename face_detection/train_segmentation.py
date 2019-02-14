import model
import tensorflow as tf
import numpy as np
import random

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 200

#######################################		LOAD DATA
print('Loading Data...')
x_train = np.load('images.npy')
y_train = np.load('outputs.npy')
x_test = x_train[2276: ]
y_test = y_train[2276: ]
x_train = x_train[: 2276]
y_train = y_train[: 2276]
#######################################

print('Initializing Model...')
segmentation_model = model.model()
saver = tf.train.Saver()
best_iou = 0
for epoch in range(EPOCHS):
	for batch_no in range((np.shape(x_train)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch_x = x_train[L: R]
		batch_y = y_train[L: R]
		#temp = segmentation_model.session.run([segmentation_model.intersection, segmentation_model.union], feed_dict = {segmentation_model.labels: batch_y, segmentation_model.net.input_layer: batch_x})
		#print(temp)
		#exit(0)
		segmentation_model.train_model(batch_x, batch_y, LEARNING_RATE)

	IOU_Train = 0.0
	IOU_Test = 0.0
	train_iter = 0
	test_iter = 0
	cost = 0.0

	for batch_no in range((np.shape(x_train)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		train_iter += 1
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch_x = x_train[L: R]
		batch_y = y_train[L: R]
		IOU_Train += segmentation_model.get_IOU(batch_x, batch_y)
		cost += segmentation_model.cost(batch_x, batch_y)

	for batch_no in range((np.shape(x_test)[0] + BATCH_SIZE - 1) // BATCH_SIZE):
		test_iter += 1
		L = batch_no * BATCH_SIZE
		R = L + BATCH_SIZE
		batch_x = x_test[L: R]
		batch_y = y_test[L: R]
		IOU_Test += segmentation_model.get_IOU(batch_x, batch_y)
	cost /= train_iter
	IOU_Train /= train_iter
	IOU_Test /= test_iter
	IOU_Train *= 100.0
	IOU_Test *= 100.0
	if(IOU_Test > best_iou):
		saver.save(segmentation_model.session, './model')
		best_iou = IOU_Test
	print('Epoch: {:5d}\tLoss: {:5f}\tTrain-IOU: {:5f}%\tTest-IOU: {:5f}%'.format(epoch, cost, IOU_Train, IOU_Test))

