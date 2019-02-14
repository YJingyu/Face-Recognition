import collections
import numpy as np
import cv2 as cv

def plot(image):
	plt.imshow(image)
	plt.show()
	plt.waitKey(0)
	plt.destroyAllWindows()

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
		for dx in range(-2, 3):
			for dy in range(-2, 3):
				current_x = x + dx
				current_y = y + dy
				if((dx == 0 and dy == 0) == False and current_x >= 0 and current_x < X and current_y >= 0 and current_y < Y and image[current_x][current_y] > 0):
					image[current_x][current_y] = 0
					queue.append((current_x, current_y))
	return (image, (min_x, min_y), (max_x, max_y))

def get_bounding_boxes(image):			#image has to be a 2 dimensional array
	X, Y = np.shape(image)[: 2]
	boxes = []
	for i in range(X):
		for j in range(Y):
			if(image[i][j] > 0):
				image, (min_x, min_y), (max_x, max_y) = bfs(image, i, j)
				if(max_x - min_x > 25 and max_y - min_y > 25):
					boxes.append(((min_x - 7, min_y - 7), (max_x + 7, max_y + 7)))
	final = []
	for (min_x, min_y), (max_x, max_y) in boxes:
		X = max_x - min_x
		Y = max_y - min_y
		q1 = 2 * X
		q2 = 3 * Y
		diff = q1 - q2
		if(diff < 0):
			req = Y * 3 / 2 - X
			# min_x = int(min_x - req / 2)
			max_x = int(max_x + req)
		# else:
		# 	req = X * 2 / 3 - Y
		# 	min_y = int(min_y - req / 2)
		# 	max_y = int(max_y + req / 2)
		min_x = max(min_x, 0)
		min_y = max(min_y, 0)
		max_x = min(max_x, np.shape(image)[0])
		max_y = min(max_y, np.shape(image)[1])
		final.append(((min_x, min_y), (max_x, max_y)))
	return final

def _draw_line(image, x1, y1, x2, y2, value):		#line should be vertical or horizontal
	if(x1 > x2):
		x1, x2 = (x2, x1)
	if(y1 > y2):
		y1, y2 = (y2, y1)
	x2 = min(x2, np.shape(image)[0] - 1)
	y2 = min(y2, np.shape(image)[1] - 1)
	if(x1 > x2):
		return image
	if(y1 > y2):
		return image
	x1 = max(x1, 0)
	y1 = max(y1, 0)
	dx = (x2 - x1) // max(1, abs(x2 - x1))
	dy = (y2 - y1) // max(1, abs(y2 - y1))

	image[x1][y1] = value
	while(((x1 == x2) and (y1 == y2)) == False):
		x1 += dx
		y1 += dy
		image[x1][y1] = value
	return image

def draw_bounding_boxes(image, boxes, value, thickness = 4):
	for box in boxes:
		(x1, y1), (x2, y2) = box
		for _ in range(thickness):
			image = _draw_line(image, x1 - 5, y1 - 5, x2 + 5, y1 - 5, value)
			image = _draw_line(image, x2 + 5, y1 - 5, x2 + 5, y2 + 5, value)
			image = _draw_line(image, x2 + 5, y2 + 5, x1 - 5, y2 + 5, value)
			image = _draw_line(image, x1 - 5, y2 + 5, x1 - 5, y1 - 5, value)
			x1 += 1
			y1 += 1
			x2 -= 1
			y2 -= 1
	return image
