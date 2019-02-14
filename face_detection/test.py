import model
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 200

#######################################		LOAD DATA
print('Loading Data...')
x_test = np.load('images.npy')[2276: ]
#######################################

print('Initializing Model...')
segmentation_model = model.model('./saved_model/model')
plt.gray()
indices = random.sample(range(0, len(x_test)), 10)
for index in indices:
	image = x_test[index]
	gray = cv.cvtColor(np.uint8(image), cv.COLOR_BGR2GRAY)
	output = segmentation_model.segment([image])
	gray = np.asarray(gray).reshape((200, 200))
	output = np.asarray(output).reshape((200, 200))
	final = np.concatenate((gray, output), axis = 1)
	plt.imshow(final)
	cv.imwrite(str(index) + '.jpg', output)
	plt.show()