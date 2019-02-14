import cv2
import os
from itertools import product
import sys
import numpy as np
import tensorflow as tf
sys.path.append('face_detection')
sys.path.append('face_recognition')
import detect
import recognize
import get_bounding_boxes as box

detector = detect.seg()
recognizer = recognize.recog()
var_list1 = [detector.net.wconv1, detector.net.bconv1, detector.net.wconv2, detector.net.bconv2, detector.net.wconv3, detector.net.bconv3, detector.net.wconv4, detector.net.bconv4, detector.net.wconv5, detector.net.bconv5, detector.net.wconv6, detector.net.bconv6, detector.net.wconv7, detector.net.bconv7, detector.net.wconv8, detector.net.bconv8, detector.net.wconv9, detector.net.bconv9]
var_list2 = [recognizer.encoder.wconv1, recognizer.encoder.bconv1, recognizer.encoder.wconv2, recognizer.encoder.bconv2, recognizer.encoder.wconv3, recognizer.encoder.bconv3, recognizer.encoder.wconv4, recognizer.encoder.bconv4, recognizer.encoder.wconv5, recognizer.encoder.bconv5, recognizer.encoder.wfc_encoder, recognizer.encoder.bfc_encoder]

def process(image):
	print(np.shape(image))
	X, Y = np.shape(image)[: 2]
	kernel_size = 30
	# x_steps = (X + kernel_size - 1) // kernel_size
	# y_steps = (Y + kernel_size - 1) // kernel_size
	x_steps = X - kernel_size + 1
	y_steps = Y - kernel_size + 1
	final_image = np.copy(image)
	for i, j in product(range(x_steps), range(y_steps)):
		kernel = image[i: i + kernel_size, j: j + kernel_size]
		kernel = np.asarray(kernel)
		mx = np.max(kernel)
		mn = np.min(kernel)

		final_image[i, j] = (((image[i, j] - mn) / (0.0001 + mx - mn)) * 255.0).astype(int)
	return final_image
session = tf.Session()
session.run(tf.global_variables_initializer())
saver1 = tf.train.Saver(var_list1)
saver1.restore(session, detector.path)
saver2 = tf.train.Saver(var_list2)
saver2.restore(session, recognizer.path)

def process_bounding_boxes(boxes, fx, fy):
	processed_bounding_boxes = []
	for ((x1, y1), (x2, y2)) in boxes:
		processed_bounding_boxes.append(((int(x1 * fx), int(y1 * fy)), (int(x2 * fx), int(y2 * fy))))
	return processed_bounding_boxes

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

def get_faces(image, boxes):
	image = np.asarray(image)
	faces = np.zeros(shape = [0, 64, 64, 3])
	for ((x1, y1), (x2, y2)) in boxes:
		temp = image[x1: x2, y1: y2,:]
		if(np.size(temp)):
			temp = np.asarray(cv2.resize(temp, (64, 64), interpolation = cv2.INTER_CUBIC)).reshape([1, 64, 64, 3])
			faces = np.concatenate((faces, temp))
	identity = recognizer.get_result(faces, session)
	return identity

while True:
	ret, frame = cam.read()
	if not ret:
		break
	#######################################################################
	# img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	# frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	# frame = process(frame)
	mn = np.min(frame)
	mx = np.max(frame)
	frame = (((frame - mn) / (0.0001 + mx - mn)) * 255.0).astype('uint8')
	#######################################################################
	k = cv2.waitKey(1)
	if k%256 == 27:
		break
	else:
		# SPACE pressed
		X, Y = np.shape(frame)[: 2]
		fx = X / 200.0
		fy = Y / 200.0
		img_name = "opencv_frame_{}.png".format(img_counter)
		resized_frame = np.asarray(cv2.resize(frame, (200, 200), interpolation = cv2.INTER_CUBIC))
		segmented_frame = detector.segment(resized_frame.reshape([1, 200, 200, 3]), session)
		##################
		copy = np.copy(segmented_frame)
		boxes = box.get_bounding_boxes(segmented_frame)
		resized_boxes = process_bounding_boxes(boxes, fx, fy)
		identities = get_faces(frame, resized_boxes)
		
		for box_, identity in zip(resized_boxes, identities):
			if identity == True:
				frame = box.draw_bounding_boxes(frame, [box_], [0, 255, 0])
			else:
				frame = box.draw_bounding_boxes(frame, [box_], [0, 0, 255])
		cv2.imshow("Face Detector", frame)
		# cv2.imwrite(img_name, frame)
		# print("{} written!".format(img_name))
		img_counter += 1

cam.release()
cv2.destroyAllWindows()