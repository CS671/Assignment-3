import tensorflow as tf
import numpy as np
import glob
import math
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
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

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
#import h5py
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from shutil import copyfile
import shutil

#from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed = 7
np.random.seed(seed)

try:
    shutil.rmtree("coordinate_file")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("coordinate_file")

y_train =[]

lineList = list()
for file in sorted(glob.glob('/home/dlagroup8/Assignment3/Q3/Ground_truth/*txt')):
    y = np.loadtxt(file)
    y_train.append(y)
y_train = np.asarray(y_train)

   # return x_train, y_train
print(y_train.shape)


img_1 = np.zeros((600,400,3))
# load data from the path specified by the user
def data_loader(path_train,y_train):
    train_list = os.listdir(path_train)
    num_classes = len(train_list)

#    # Empty lists for loading training and testing data images as well as corresponding labels

    x_train = []
  #  y_test = []
    i=0
    j=0
    # Loading training data
    for label, elem in enumerate(train_list):

        path1 = path_train + '/' + str(elem)
        images = os.listdir(path1)
        for elem2 in sorted(images):
            x_addition=0
            y_addition=0
            path2 = path1 + '/' + str(elem2)
            # Read the image form the directory
            img = cv2.imread(path2)
            img_1 = np.pad(img, ((math.ceil((600 - img.shape[0]) / 2), math.ceil((600 - img.shape[0]) / 2)),
                                 (math.ceil((400 - img.shape[1]) / 2), math.ceil((400 - img.shape[1]) / 2)), (0, 0)),
                           'edge')
            x_addition=int(math.ceil((600 - img.shape[0]) / 2))
            print(x_addition)

            y_addition=int(math.ceil((400 - img.shape[1]) / 2))
            print(y_addition)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(y_train[i][j])
            print(y_train[i][j+1])
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            y_train[i][j]=int(y_train[i][j])+x_addition
            y_train[i][j+1]=int(y_train[i][j+1])+y_addition
            print(y_train[i][j])
            print(y_train[i][j + 1])
            print(i)
            i+=1
            j=0


#            # Append image to the train data list
            x_train.append(img_1)
   #         print(path2)
#           
    x_train = np.asarray(x_train)

    return x_train,y_train
    # Convert lists into numpy arrays

print(">>" + os.path.basename(__file__) + " -- phase = test -- ")
path_train = input("Enter the training path for testing:") 


#*********************************************************
path = "./Assignment3/Q3/Core_Point/Data"

files = os.listdir(path)
files.sort()
filename =[]

for f in files:
	filename.append(f.split(".")[0])



bechain_dil = np.asarray(filename)





# define baseline model
X_train1 ,y_train1= data_loader(path_train,y_train)

X_train, X_test, y_train, y_test,bechain_dil_train,bechain_dil_test = train_test_split(X_train1, y_train1,bechain_dil, test_size=0.33, random_state=42)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print(X_train.shape)
print(y_test[3])





# normalize inputs from 0-255 to 0-1
X_test = X_test
y_test = y_test
#y_train = np.asarray(y_train)



# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32

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
    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x
# build the model


inputShape = (600, 400, 3)
inputs = Input(shape=inputShape)
model = baseline_model(inputs)
finger_print_classification = Dense(2, activation = 'relu', name='finger_print')(model)


losses = {
    "finger_print": 'mae',
}
lossWeights = {"finger_print": 1.0}

designed_model = Model(inputs, [finger_print_classification])

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
filename = "/home/dlagroup8/Model_q_3/weights-improvement-90-51.3694.hdf5"
designed_model.load_weights(filename)
designed_model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=['accuracy'])
designed_model.summary()

scores = designed_model.evaluate(X_test, y_test, verbose=1)

ypred = designed_model.predict(X_test)
for i in range(len(X_test)):
    print("X=%s, Predicted=%s" % (X_test[i], ypred[i]))
    f= open("./coordinate_file/img%s.txt"%bechain_dil[i],"w+")
    f.write(str(ypred[i]))
    f.close()
