import collections
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def plot(image):
	cv.imshow('', image)
	cv.waitKey(0)
	cv.destroyAllWindows()

def bfs(image, source_x, source_y):			#returns (image, min_corner, max_corner)
	X, Y = np.shape(image)[: 2]
	queue = collections.deque()
	queue.append((source_x, source_y))
	image[source_x][source_y] = 0
	min_x, min_y, max_x, max_y = [X + 1, Y + 1, -1, -1]
	while(len(queue) > 0):
		node = queue.popleft()
		x, y = node
		min_x = min(min_x, x)
		max_x = max(max_x, x)
		min_y = min(min_y, y)
		max_y = max(max_y, y)
		# min_x, min_y = min((min_x, min_y), node)
		# max_x, max_y = max((max_x, max_y), node)
		for dx, dy in zip([-1, 1, 0, 0], [0, 0, 1, -1]):
			current_x = x + dx
			current_y = y + dy
			if((dx == 0 and dy == 0) == False and current_x >= 0 and current_x < X and current_y >= 0 and current_y < Y and image[current_x][current_y] > 0):
				image[current_x][current_y] = 0
				queue.append((current_x, current_y))
	return (image, (min_x, min_y), (max_x, max_y))

def get_bounding_boxes(image_):			#image has to be a 2 dimensional array
	image = np.copy(image_)
	X, Y = np.shape(image)[: 2]
	boxes = []
	for i in range(X):
		for j in range(Y):
			if(image[i][j] > 0):
				image, (min_x, min_y), (max_x, max_y) = bfs(image, i, j)
				if(min_x < max_x - 20 and min_y < max_y - 20):
					boxes.append(((min_x, min_y), (max_x, max_y)))
	return boxes

def _draw_line(image, x1, y1, x2, y2):		#line should be vertical or horizontal
	if(len(np.shape(image)) > 2):
		value = [0, 0, 255]
	else:
		value = 255
	dx = (x2 - x1) // max(1, abs(x2 - x1))
	dy = (y2 - y1) // max(1, abs(y2 - y1))
	image[x1][y1] = value
	while((x1 == x2 and y1 == y2) == False):
		x1 += dx
		y1 += dy
		image[x1][y1] = value

def draw_bounding_boxes(image, boxes = None, thickness = 4):
	for box in boxes:
		(x1, y1), (x2, y2) = box
		_draw_line(image, x1, y1, x2, y1)
		_draw_line(image, x2, y1, x2, y2)
		_draw_line(image, x2, y2, x1, y2)
		_draw_line(image, x1, y2, x1, y1)