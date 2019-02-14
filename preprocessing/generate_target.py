class ellipse_op:
	def inside_ellipse(self, x, y, ellipse):	#ellipse = (X, Y, theta, a, b)
		import math
		b, a, theta, Y, X = ellipse
		theta = -theta
		xx = (x - X) * math.cos(theta) + (y - Y) * math.sin(theta)
		yy = -(x - X) * math.sin(theta) + (y - Y) * math.cos(theta)
		return xx * xx / (a * a) + yy * yy / (b * b) <= 1.0

	def generate_target(self, image, ellipse_list):		#(x_center, y_center, r_major, r_minor, theta):
		import numpy as np
		shape = np.shape(image)
		target_map = np.zeros(shape = shape[: 2], dtype = int)
		for i in range(shape[0]):
			for j in range(shape[1]):
				check = (max(map(int, [self.inside_ellipse(i, j, ellipse) for ellipse in ellipse_list])) > 0)
				if(check):
					target_map[i][j] = 255
		return target_map
