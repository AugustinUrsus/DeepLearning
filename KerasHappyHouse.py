import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
from keras.layers import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('hello world')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))



def HappyModel(input_shape):
    """
        Implementation of the HappyModel.
        
        Arguments:
        input_shape -- shape of the images of the dataset
        
        Returns:
        model -- a Model() instance in Keras
        """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    
    X_input = Input(input_shape)
        
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32,(7,7), strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    ### END CODE HERE ###
    
    return model

print("X_train shape(1:4): " + str(X_train.shape[1:4]))
happyModel = HappyModel(X_train.shape[1:4])


### START CODE HERE ### (1 line)

happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
happyModel.fit(x = X_train, y = Y_train, epochs = 20, batch_size = 50)
preds = happyModel.evaluate(X_test, Y_test, batch_size = 32, verbose = 1, sample_weight = None)
print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)


print(happyModel.predict(x))

img_path = 'images/my_image01.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)


print(happyModel.predict(x))

img_path = 'images/my_image02.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

y = image.img_to_array(img)
y = np.expand_dims(y, axis = 0)
y = preprocess_input(y)


print(happyModel.predict(y))

img_path = 'images/my_image03.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

z = image.img_to_array(img)
z = np.expand_dims(z, axis = 0)
z = preprocess_input(z)


print(happyModel.predict(z))

img_path = 'images/my_image04.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

a = image.img_to_array(img)
a = np.expand_dims(a, axis = 0)
a = preprocess_input(a)
print(happyModel.predict(a))

img_path = 'images/my_image05.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

b = image.img_to_array(img)
b = np.expand_dims(b, axis = 0)
b = preprocess_input(b)
print(happyModel.predict(b))

img_path = 'images/my_image06.jpg'
img = image.load_img(img_path, target_size = (64,64))
imshow(img)

c = image.img_to_array(img)
c = np.expand_dims(c, axis = 0)
c = preprocess_input(c)
print(happyModel.predict(c))
happyModel.summary()

