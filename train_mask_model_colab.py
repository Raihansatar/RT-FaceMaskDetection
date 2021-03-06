# Python file for training the mask model, this code is use in google colab

# import libraris per file withoud sub file
from sys import path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for, and batch siz
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = "/content/drive/MyDrive/facemask/new_dataset"
CATEGORIES = ["without_mask", "correct_mask", "chin_mask", "mouth_chin_mask", "nose_mouth_mask"]

print("[INFO] Loading images...")

data = []
labels = []
for category in CATEGORIES:
    current_image = 0
    path = os.path.join(DIRECTORY, category)
    for subfolder in os.listdir(path): # return all dir in that path
      # image_no = 0
      subpath = os.path.join(path, subfolder)
      # print(subpath)
      img_path = subpath # join path to corresponding image
      print(img_path)
      image = load_img(img_path, target_size=(224, 224))
      image = img_to_array(image) # convert image to array
      image = preprocess_input(image)

      print (str(current_image) +  " of " + str(len(os.listdir(path))) + " in " + category)
      current_image = current_image + 1
      data.append(image) # append image in data
      labels.append(category) # append category

# save the label and image data incase something crashed
np.save('/content/drive/MyDrive/facemask/data', data)
np.save('/content/drive/MyDrive/facemask/labels', labels)
# we get data as numerical value but label still not

print("Finish")

# below try to convert label to numerical perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42) 

# construct the training image generator for data augmentation Generate batches of tensor image data with real-time data augmentation.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the the base model n_labels = len(set(labels))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to fiSnd the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")

# Save the model
model.save("/content/drive/MyDrive/facemask/mask_detector_model.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# save the plotted figure
plt.savefig("/content/drive/MyDrive/facemask/plot.png")