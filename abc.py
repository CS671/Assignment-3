import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import h5py
from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.losses import sparse_categorical_crossentropy
import glob
import IntersectionOverUnion


image = []
path1 = "./Assignment3/Q1/Knuckle/Data/*.jpg"
path2 = "./Assignment3/Q1/Palm/Data/*.jpg"
path3 = "./Assignment3/Q1/Vein/Data/*.jpg"
#dataset1 = os.listdir(path1)

for file in glob.glob(path1):
    I = cv2.imread(file)
    #I=cv2.resize(I,(480,640))
    img = np.asarray(I)
    # print(I)
    # print(img.shape)
    x,y,z = img.shape
    d = 480 - x
    e = 640 - y
    img1 = np.pad(img, ([int(d/2), int(d/2)], [int(e/2), int(e/2)],[int(0), int(0)]), 'constant')
    # print(img1)
    # print(a, b)
    # print(img.shape)
    # print(img1.shape)
    image.append(img1)
'''
dataset2 = os.listdir(path2)
for file in dataset2:
    I = cv2.imread(path2 + "/" + file,0)
    img = np.asarray(I)
    # print(I)
    # print(img.shape)
    a, b = img.shape
    a = 480 - a
    b = 640 - b
    img1 = np.pad(img, ([int(a/2), int(a/2)], [int(b/2), int(b/2)]), 'constant')
    # print(img1)
    # print(a, b)
    # print(img.shape)
    # print(img1.shape)
    image.append(img1)
'''
#dataset3 = os.listdir(path3)
for file1 in glob.glob(path3):
    I1 = cv2.imread(file1)
    #I=cv2.resize(I1,(480,640))
    img = np.asarray(I1)
    # print(I1)
    # print(img.shape)
    x1,y1,z1= img.shape
    # print(a,b)
    a = 480 - x1
    b = 640 - y1
    img1 = np.pad(img, ([int(a/2), int(a/2)], [int(b/2), int(b/2)],[int(0) ,int(0)]), 'constant')
    # print(img1)
    # print(a, b)
    # print(img.shape)
    # print(img1.shape)
    image.append(img1)
image = np.asarray(image)
print(image)



f=open('./Assignment3/Q1/Knuckle/groundtruth.txt')
f1=open('./Assignment3/Q1/Palm/groundtruth.txt')
f2=open('./Assignment3/Q1/Vein/groundtruth.txt')
lines_1=f.readlines()
lines_2=f1.readlines()
lines_3=f2.readlines()
lines_1.sort()
lines_2.sort()
lines_3.sort()
name=[]
bbox1=[]
bbox2=[]
bbox3=[]
bbox4=[]
cat=[]
for x in lines_1:
    name.append(x.split(',')[0])
    bbox1.append(x.split(',')[1])
    bbox2.append(x.split(',')[2])
    bbox3.append(x.split(',')[3])
    bbox4.append(x.split(',')[4])
    cat.append(0)
'''    
for x in lines_2:
    name.append(x.split(',')[0])
    bbox1.append(x.split(',')[1])
    bbox2.append(x.split(',')[2])
    bbox3.append(x.split(',')[3])
    bbox4.append(x.split(',')[4])
    cat.append(x.split(',')[5])
'''        
for x in lines_3:
    name.append(x.split(',')[0])
    bbox1.append(x.split(',')[1])
    bbox2.append(x.split(',')[2])
    bbox3.append(x.split(',')[3])
    bbox4.append(x.split(',')[4])
    cat.append(1)

list = bbox1 + bbox2 + bbox3 + bbox4
list = np.asarray(list)
list = np.reshape(list, [4, 6960])
list = np.transpose(list)

name=np.asarray(name)
bbox1=np.asarray(bbox1)
bbox2=np.asarray(bbox2)
bbox3=np.asarray(bbox3)
bbox4=np.asarray(bbox4)
cat=np.asarray(cat)



print(name.shape)    
#f.close()
seed = 7
np.random.seed(seed)


# load data from the path specified by the user



X_train, X_test, y_train_bbox1, y_test_bbox1,y_train_bbox2, y_test_bbox2,y_train_bbox3, y_test_bbox3,y_train_bbox4, y_test_bbox4,y_train_cat, y_test_cat, gt_train, gt_test = train_test_split(image,bbox1,bbox2,bbox3,bbox4,cat, list, test_size=0.25)
#X_train=X_train/255
X_train=np.asarray(X_train)
y_train_bbox1=np.asarray(y_train_bbox1)
y_train_bbox2=np.asarray(y_train_bbox2)
y_train_bbox3=np.asarray(y_train_bbox3)
y_train_bbox4=np.asarray(y_train_bbox4)
#y_train_cat=y_train_cat/640
y_train_cat=np.asarray(y_train_cat)
y_train_cat=np_utils.to_categorical(y_train_cat)
y_test_cat=np.asarray(y_test_cat)
y_test_cat=np_utils.to_categorical(y_test_cat)

#gt_train=np.divide(gt_train,640)
gt_train=np.asarray(gt_train)
gt_test=np.asarray(gt_test)
print(X_train)
print(y_train_bbox1.shape)
print(y_train_bbox1)


EPOCHS = 30
INIT_LR = 1e-3
BS = 16
IMAGE_DIMS = (480,640)
def baseline_model(inputs):
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.5)(x)

    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
# CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)


    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)


    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    return x

inputShape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
inputs = Input(shape=inputShape)
model = baseline_model(inputs)
head_classification = Dense(2, activation='softmax', name='class_output')(model)

head_regression = Dense(4, activation='relu', name='reg_output')(model)
losses = {
	"class_output": "binary_crossentropy",
	"reg_output": "mae",
    
}
lossWeights = {"class_output": 0.6, "reg_output": 0.4}

designed_model = Model(inputs, [head_classification,head_regression])

#callback_1 = keras.callbacks.TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True, write_images=True)
filepath="./Model_Assignment_3.2/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
designed_model.compile(optimizer='nadam', loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])
# train the network to perform multi-output classification
designed_model.summary()
designed_model.fit(X_train,
	{"class_output": y_train_cat,"reg_output":gt_train},validation_data=(X_test, {"class_output": y_test_cat,"reg_output":gt_test}),
	epochs=EPOCHS,batch_size=BS,
	verbose=1,callbacks=callbacks_list)

scores = designed_model.evaluate(X_test, {"class_output": y_test_cat,"reg_output": gt_test }, verbose=1)
#scores1 = designed_model.evaluate(X_test, {"class_output": y_test_cat}, verbose=1)
print(scores)

#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
ypred = designed_model.predict(X_test)
#print(ypred.shape)
print("X=%s, Predicted=%s" % (X_test, ypred))
print("**********************************")
print(" actual=%s" %(gt_test[10,:]))

