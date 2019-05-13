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
from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
#from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"]="1"
seed = 7
np.random.seed(seed)


y_train =[]

lineList = list()
for file in sorted(glob.glob('/home/dlagroup8/Assignment3/Q3/Ground_truth/*txt')):
    y = np.loadtxt(file)
    y_train.append(y)
y_train = np.asarray(y_train)

   # return x_train, y_train
print(y_train.shape)


#for file in sorted(glob.glob('/home/souvik/Desktop/DEEP_learning/Assignment3/Q3/Core_Point/Data/*jpeg')):
 #   img = cv2.imread(file)
#    x_train.append(img)
# fix random seed for reproducibility
# we always initialize the random number generator to a constant seed #value for reproducibility of results.


img_1 = np.zeros((600,400,3))
# load data from the path specified by the user
def data_loader(path_train,y_train):
    train_list = os.listdir(path_train)
 #   '''
    # Map class names to integer labels
#    train_class_labels = { label: index for index, label in enumerate(class_names) }
#    '''
#    # Number of classes in the dataset
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
#            # Append class-label corresponding to the image
#            y_train.append(str(lineList[0]))

    x_train = np.asarray(x_train)

    return x_train,y_train
    # Convert lists into numpy arrays


EPOCHS = 100
INIT_LR = 1e-3
BS = 32

print(">>" + os.path.basename(__file__) + " -- phase = train -- " + "epochs = %s " % Epochs )


path_train = input("Enter the path:")

# define baseline model
X_train ,y_train= data_loader(path_train,y_train)



print(X_train.shape)

#input_shape = (X_train.shape[0], X_train.shape[1])
# forcing the precision of the pixel values to be 32 bit
#X_train = X_train.astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train
y_train = y_train
#y_train = np.asarray(y_train)



# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


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
designed_model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=['accuracy'])

#callback_1 = keras.callbacks.TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True, write_images=True)
filepath="./Model_q_3/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# train the network to perform multi-output classification
designed_model.summary()
designed_model.fit(X_train,
	{"finger_print": y_train},
	epochs=EPOCHS,batch_size=BS,
	verbose=1,callbacks = callbacks_list)
