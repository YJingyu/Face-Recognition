import os
import numpy as np
import cv2 as cv
cnt = 0
#####################################################
#This is going to save the images and labels as a numpy array
#Run this script only once
#Complete preprocessing has been done- No need to reshape at any step
#####################################################
x = np.asarray([]).reshape([0, 200, 200, 3])
y = np.asarray([]).reshape([0, 200, 200, 1])
for root, directories, files in os.walk('originalPics'):
	for file in files:
		path = os.path.join(root, file)
		path_x = path
		path_y = ('outputs' + path.split('originalPics')[1])[: -4] + 'target.jpg'
		if(os.path.isfile(path_y)):
			print(path_x + ' <---> ' + path_y)
			X = cv.imread(path_x)
			Y = cv.imread(path_y)
			Y = cv.cvtColor(Y, cv.COLOR_BGR2GRAY)
			X = cv.resize(X, (200, 200), interpolation = cv.INTER_CUBIC)
			Y = cv.resize(Y, (200, 200), interpolation = cv.INTER_CUBIC)
			x = np.concatenate((x, X.reshape([1, 200, 200, 3])), axis = 0)
			y = np.concatenate((y, Y.reshape([1, 200, 200, 1])), axis = 0)

print(np.shape(x))
print(np.shape(y))
np.save('images.npy', x)
np.save('outputs.npy', y)