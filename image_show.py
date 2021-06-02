from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os


DIRECTORY = r"C:\Users\RaihanSatar\OneDrive\Desktop\Soft Computing Face Recognition\dataset"
CATEGORIES = ["correct_mask", "incorrect_mask"]



def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))

			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
        
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detectorv3.model")

print("[INFO] starting reading file...")


for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    loop = 1
    for img in os.listdir(path): # return all dir in that path
        if loop > 50:
            break
        img_path = os.path.join(path, img) # join path to corresponding image
        print(img_path)

        # read image
        image = cv2.imread(img_path)
        # image = imutils.resize(image, width=400)
        # print("Read Image")
        # detect faces in the image and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, incorrect_mask, without_mask,) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            # label = "Mask" if mask > incorrect_mask else "Incorrect Mask"
            if mask > incorrect_mask and mask > without_mask:
                label = "Mask"
                print ("Maskqweqweqwe:" , mask, "Incorrect Mask:" , incorrect_mask, " W/Mask:" , without_mask)
            elif incorrect_mask > without_mask:
                label = "Incorrect Mask"
                print ("Mask:" , mask, "Incorrect Maskqweqweqwe:" , incorrect_mask, " W/Mask:" , without_mask)
            else:
                label = "Without Mask"
                print ("Mask:" , mask, "Incorrect Mask:" , incorrect_mask, " W/Maskqweqweqwe:" , without_mask)
            
            # print ("Mask:" , mask, "Without Mask:" , incorrect_mask) if mask > incorrect_mask else print ("without Mask:" , incorrect_mask, "Mask: ", mask)
            # print ("Mask:" , mask, "Incorrect Mask:" , incorrect_mask, " W/Mask:" , without_mask)
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label == "Mask":
                color = (0, 255, 0)
            elif label == "Incorrect Mask":
                (0, 0, 255)
            else:
                (255, 0, 0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, incorrect_mask, without_mask) * 100)

            # display the label and bounding box rectangle on the output
            # image
            image = cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        # cv2.imshow("image", image)
        # cv2.waitKey(0) & 0xFF

        filename = "output_picture/"+ category +"/image_"+ str(loop)+ ".jpg"
        # print(filename)
        cv2.imwrite(filename, image)
        loop = loop + 1
        # print (loop)





    #     image = load_img(img_path, target_size=(224, 224))
    #     image = img_to_array(image) # convert image to array
    #     image = preprocess_input(image) # mobile app related??

    #     data.append(image) # append image in data
    #     labels.append(category) # append category

# path = r'dataset\correct_mask\00000_Mask.jpg'

# path = os.listdir()


# image = cv2.imread(path, 0)
# cv2.imshow('image', image)
# key = cv2.waitKey(0) & 0xFF
# if key == ord("q"):
#     cv2.destroyAllWindows()
