## Loading Data
1. `wget <link>` command can be used to download the data into cloud.
2. Use package cv2 to load image file (see the example).

#### Example of package cv2
```
import cv2

src = cv2.imread('/home/ex.png', cv2.IMREAD_UNCHANGED)

# percent by which the image is resized
scale_percent = 50

# calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

#dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)

cv2.imwrite('/home/resize_ex.png', output)
```
- `cv2.imread(path)` reads the given file in `cv2.IMREAD_UNCHANGED` and returns a numpy array
- `scr.shape[1]` gives the width of the source image.
- `cv2.resize(scr, dsize)` resize the image `src` to the size `dsize` and returns numpy array.
- `cv2.imwrite(path)` writes the output to a file.

Source: [Python OpenCV cv2 Resize Image](https://pythonexamples.org/python-opencv-cv2-resize-image/)

3. `LabelEncoder()` can encode target lables with value between 0 and n_classes-1. Pay attention: This transformer should be used to encode target value, i.e. y, and not the input, x.

#### Example of LabelEncoder()
```
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
```
Source: [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

4. `np.save(file, arr)` can save an array to a binary file in NumPy.npy formate.

## Train Model
This part focus on how to create a Sequential model with keras.
```
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = 300
N_EPOCHS = 1000
BATCH_SIZE = 2048
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load("x_train.npy"), np.load("y_train.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Dense(N_NEURONS, input_dim=7500, activation="relu"),
    Dense(4, activation="softmax")
])
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_yucao.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
```
