import generate_target
import cv2 as cv
ellipse_op = generate_target.ellipse_op()

for index in range(1, 11):
	file_path = 'FDDB-folds/FDDB-fold-' + str(index).zfill(2) + '-ellipseList.txt'
	print('-->Processing File: {}'.format(file_path))
	file = open(file_path, 'r')
	info = file.read().split('\n')
	ptr = 0
	while(ptr < len(info)):
		if(len(info[ptr: ]) <= 1):
			break
		file_name, n = info[ptr: ptr + 2]
		n = int(n)
		ptr += 2
		ellipse_list =  [ellipse.split() for ellipse in info[ptr: ptr + n]]
		ellipse_list = [[float(i) for i in ellipse][: -1] for ellipse in ellipse_list]
		ptr += n		#skip the 1 at the end of each entry
		#load image
		image = cv.imread('outputs/' + file_name + '.jpg')
		target = ellipse_op.generate_target(image, ellipse_list)
		print('outputs/' + file_name + '.jpg --> ' + 'outputs/' + file_name + 'target.jpg')
		cv.imwrite('outputs/' + file_name + 'target.jpg', target)
	file.close()