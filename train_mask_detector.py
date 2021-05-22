# import libraris
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

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\RaihanSatar\OneDrive\Desktop\Soft Computing Face Recognition\dataset"
CATEGORIES = ["correct_mask", "incorrect_mask"]

# dir_list = os.listdir(DIRECTORY)  
# print("Files and directories in '", DIRECTORY, "' :") 
  
# # print the list
# print(dir_list)


print("[INFO] Loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path): # return all dir in that path
        img_path = os.path.join(path, img) # join path to corresponding image
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image) # convert image to array
        image = preprocess_input(image) # mobile app related??

        data.append(image) # append image in data
        labels.append(category) # append category


# we get data as numerical value but label still not
# below try to convert label to numerical
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42) # can be modified