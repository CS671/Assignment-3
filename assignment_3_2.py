from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import os
import cv2
#from PIL import Image
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, BatchNormalization,Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.environ["CUDA_VISIBLE_DEVICES"]="1"

seed = 7
np.random.seed(seed)

data_path = "./Assignment3/Q2/Data"
mask_path = "./Assignment3/Q2/Mask"


def get_data(data_path, mask_path):
    data = []
    for frame in os.listdir(data_path):
        im_data = cv2.imread(os.path.join(data_path, frame))
        #im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        im_data = cv2.resize(im_data, (256, 256))
        data_array = np.array(im_data)
        data.append(data_array)

    data = np.asarray(data)
    #data = data/255

    mask = []
    for f in os.listdir(mask_path):
        im_mask = cv2.imread(os.path.join(mask_path, f),0)
        #im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
        im_mask = cv2.resize(im_mask, (256, 256))
        mask_array = np.array(im_mask)
        mask.append(mask_array)

    mask = np.asarray(mask)
    mask = mask/255

    return data, mask


data, mask = get_data(data_path, mask_path)
print(data.shape)
print(mask.shape)
data_train, data_test, mask_train, mask_test = train_test_split(data, mask, test_size=0.1)

def image_generator(x_train,y_train, batch_size = 32):

    count= -batch_size
    y_train=np.expand_dims(y_train, axis=3)
    while True:
        # Select files (paths/indices) for the batch
        count+=batch_size
        if(count>=9000-batch_size):
            count=0
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for i in range(batch_size):
            input1 = x_train[i+count]
            output = y_train[i+count]

            # input = preprocess_input(image=input)
            batch_input += [ input1 ]
            batch_output += [ output ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        #print(batch_x.shape,batch_y.shape)
        yield( batch_x, batch_y )
        
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
# first layer
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
	padding="same")(input_tensor)
	if batchnorm:
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
# second layer
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
	padding="same")(x)
	if batchnorm:
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
	return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
# contracting path
	c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
	p1 = MaxPooling2D((2, 2)) (c1)
	p1 = Dropout(dropout*0.5)(p1)
	c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
	p2 = MaxPooling2D((2, 2)) (c2)
	p2 = Dropout(dropout)(p2)
	c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
	p3 = MaxPooling2D((2, 2)) (c3)
	p3 = Dropout(dropout)(p3)
	c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
	p4 = Dropout(dropout)(p4)
	c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
	# expansive path
	u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
	u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
	u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	u8 = Dropout(dropout)(u8)
	c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
	u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	u9 = Dropout(dropout)(u9)
	c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
	model = Model(inputs=[input_img], outputs=[outputs])
	return model
	
input_img = Input((256, 256, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


batch_size = 32

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph_UNET', histogram_freq=0, write_graph=True, write_images=True)

#data_train = np.squeeze(data_train, axis=3)
#mask_train = np.squeeze(mask_train, axis=3)

trainGen = image_generator(x_train=data_train, y_train=mask_train, batch_size=batch_size)

model.fit_generator(trainGen, epochs=10, steps_per_epoch=batch_size, callbacks=[tbCallBack])
print(data_test.shape,mask_test.shape)
mask_test = np.expand_dims(mask_test, axis=3)
score = model.evaluate(data_test, mask_test)
print(score)

#data_test = np.expand_dims(data_test, axis=3)
y_pred = model.predict(data_test)
print('y_pred.shape=' + str(y_pred.shape))
print(y_pred[0])
for i in range(10):
    cv2.imwrite('output3/'+str(i)+'_pred.jpg', y_pred[i]*255)
    cv2.imwrite('output3/'+str(i)+'_input.jpg', data_test[i]*255)
    cv2.imwrite('output3/'+str(i)+'_gt.jpg', mask_test[i]*255)

