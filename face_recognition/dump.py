import os
import cv2 as cv

os.chdir('my_train_images')
ii = 0
for i in os.listdir():
	image = cv.imread(i)
	image_flipped = cv.flip(image, 1)
	cv.imwrite('final/' + str(ii) + '.jpg', image)
	cv.imwrite('final/' + str(ii + 1) + '.jpg', image_flipped)
	ii += 2