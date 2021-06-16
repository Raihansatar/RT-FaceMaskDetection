

# Python file for detect the picture as input

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):

		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))

			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:

		# for faster inference we'll make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
        
	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector_model.model")

print("[INFO] starting reading file...")

DIRECTORY = r"C:\Users\RaihanSatar\OneDrive\Desktop\Soft Computing Face Recognition\testing_"

loop = 1
for file in os.listdir(DIRECTORY): 
	path = os.path.join(DIRECTORY, file)
	
	# grab the image from the threaded video stream and resize it to have a maximum width of 400 pixels
	image = cv2.imread(path)

	# detect/predict faces in the image and determine if they are wearing a face mask or not
	(locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
	# loop over the detected face locations and their corresponding locations

	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box

		(chin_mask, correct_mask, mouth_chin_mask, nose_mouth_mask, without_mask) = pred

		# determine the class label and color we'll use to draw the bounding box and text
		if correct_mask > without_mask and correct_mask > chin_mask and correct_mask > mouth_chin_mask and correct_mask > nose_mouth_mask:
			label = "Correct Mask"
		elif chin_mask > without_mask and chin_mask > correct_mask and chin_mask > mouth_chin_mask and chin_mask > nose_mouth_mask:
			label = "Incorrect Mask - Chin Only"
		elif mouth_chin_mask > without_mask and mouth_chin_mask > chin_mask and mouth_chin_mask > correct_mask and mouth_chin_mask > nose_mouth_mask:
			label = "Incorrect Mask - Mouth And Chin Only"
		elif nose_mouth_mask > without_mask and nose_mouth_mask > chin_mask and nose_mouth_mask > mouth_chin_mask and nose_mouth_mask > correct_mask:
			label = "Incorrect Mask - Mouth and Nose Only"
		else:
			label = "No Mask"
		
		# Set the color of label and frame
		if label == "Correct Mask":
			color = (0, 255, 0)
		elif label == "Incorrect Mask - Chin Only":
			color = (156, 200, 255)
		elif label == "Incorrect Mask - Mouth And Chin Only":
			color = (155, 155, 155)
		elif label == "Incorrect Mask - Mouth and Nose Only":
			color = (0, 255, 255)
		else:
			color = (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(without_mask, correct_mask, chin_mask, mouth_chin_mask, nose_mouth_mask) * 100)

		# display the label and bounding box rectangle on the output image
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# create file name path for the output
	filename = "output_picture/image_"+ str(loop)+ ".jpg"
	
	# write the file output (image)
	cv2.imwrite(filename, image)

	# increase the loop for the next iteration
	loop = loop + 1

cv2.destroyAllWindows()
# vs.stop()