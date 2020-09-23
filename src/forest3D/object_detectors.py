'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy, opencv-python

    Functions for calling a object detection models (e.g. darknet yolo) into python.

'''

import numpy as np
import cv2

def detectObjects_yolov3(img,addr_weights,addr_confg,MIN_CONFIDENCE=0.5):
	'''

	:param img: uint 3-channel array, range 0-255. R,G,B
	:param addr_weights:
	:param addr_confg:
	:param MIN_CONFIDENCE:
	:return: returns bounding boxes as np.array Nx4:  [ymin xmin ymax xmax] in proportions
	'''

	net = cv2.dnn.readNet(addr_weights,addr_confg)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=False, crop=False)

	net.setInput(blob)
	outputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []

	for output in outputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > MIN_CONFIDENCE:
				(centerX, centerY, width, height) = detection[:4]
				boxes.append([centerY-(height/2.0),centerX-(width/2.0),centerY+(height/2.0),centerX+(width/2.0)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	return img,np.array(boxes),np.array(classIDs),np.array(confidences)