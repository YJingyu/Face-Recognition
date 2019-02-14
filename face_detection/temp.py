import bounding_box
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
image = sys.argv[1] + '.jpg'
image = cv.imread(image)
image = np.asarray(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
bounding_box.plot(image)
boxes = bounding_box.get_bounding_boxes(image)
print(boxes)
bounding_box.draw_bounding_boxes(image, boxes)
bounding_box.plot(image)