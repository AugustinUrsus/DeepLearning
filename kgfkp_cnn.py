
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.layers import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imshow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('hello world')

def load(test = False, cols = None):

    fname = 'test.csv' if test else "training.csv"
    
    df = pd.read_csv(fname)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    if cols:
        df = df[list(cols)+['Image']]

    df = df.dropna()
    columns = df.columns
    X = np.vstack(df['Image'].values)
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        X, y = shuffle(X, y, random_state = 20)
        y = y.astype(np.float32)
    else:
        y = None
        columns = None
    return X, y, columns

def load2d(test = False, cols = None):
    X, y, columns = load(test, cols)
    X = X.reshape(-1, 96, 96, 1)

    return X, y, columns

X, y, columns = load2d(test = False, cols = None)
columns = np.array(list(columns[:-1]))

#create model
def faceModel(shape):
    model = Sequential()

    model.add(BatchNormalization(input_shape = shape))
    model.add(Conv2D(24, 5, data_format = "channels_last", kernel_initializer = "he_normal", input_shape = shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))

    model.add(Conv2D(36, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(48, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(90))
    model.add(Activation('relu'))

    model.add(Dense(30))

    return model

model = faceModel((96,96,1))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)
model.fit(x = X, y = y, validation_split = 0.2, epochs = 2000, callbacks=[checkpointer], verbose=1)
model_json = model.to_json()
with open("face_model.json", "w") as json_file:
    json_file.write(model_json)

model.load_weights('face_model.h5')
y_test = model.predict(X_test)

id_lookup_frame = pd.read_csv("IdLookupTable.csv")
val_reqrd = id_lookup_frame[["ImageId", "FeatureName"]]


y_res = []
j = 0
k = 0
m = 0
for i in range(1, 1784):
    img = np.array(val_reqrd[val_reqrd["ImageId"]==i])
    if img.shape[0] == 30:
        y_res += list(y_test[i-1,:])
        j += 30
    else:
        slice_img = y_test[i-1, :]
        y_res += [slice_img[j] for j in range(30) if columns[j] in img[:, 1]]
        k += len([slice_img[j] for j in range(30) if columns[j] in img[:, 1]])
y_res = np.array(y_res)

result_dict = {
    "RowId": range(1,27125),
    "Location": y_res%96
}
result_df = pd.DataFrame(result_dict, )
result_df.to_csv("result.csv", index=False, columns=["RowId", "Location"])

