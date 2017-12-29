from __future__ import print_function
from flask import Flask, jsonify, request
import base64
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)

@app.route('/cloud1', methods=['POST'])
def get_tasks1():
	st1 = request.get_data()
	st = str(st1)
	#print st
	image_64_decode = base64.decodestring(st)
	image_result = open('dtry.jpg', 'wb')
	image_result.write(image_64_decode)
	imagePath = '/Users/shivamarora/Desktop/Minor_Final/Pedestrian_Detection/Code/dtry.jpg'
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
	
	# show the output images
	cv2.imwrite("Before_NMS.jpg", orig)
	cv2.imwrite("After_NMS.jpg", image)
	result = str(len(rects))+' original boxes, ' + str(len(pick)) + ' after suppression.'
	return result


if __name__ == '__main__':
    app.run(debug=True)