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
This part focuses on how to create a Sequential model with keras.

A sequential model can be defined as 
```
from keras.models import Sequential
model = Sequential()
model.add()
model.add()
model.add()
```
### Input Layer
The first layer in the model must specify the shape of the input. This is the number of input attributes and is defined by the input_dim argument. This argument expects an integer. 

For example, an input layer can be defined as `Dense(Neurons, input_dim=n)`, if the model has n inputs.
### Hidden Layers
#### Layer Weight Initializers
Initializers define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializers to layers depends on the layer.
```
from tensorflow.keras import layers
from tensorflow.keras import initializers

layer = layer.Dense(
      units=64
      kernel_initializer=initializers.RandomNormal(stddev=0.01)
      bias_initializer=initializers.Zeros()
      )
 ```
 The following built-in initializers are available as part of the `td.keras.initializers` model:
 1. RandomNormal
 Initializer that generates tensors with a normal distribution.
 ```
 tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
 ```
 where **mean** means mean of the random values to generate, **stddev** means standard deviation. And setting **seed** can produce the same random tensor.
 2. RandomUniform
 Initializer that generates tensors with a uniform distribution.
 ```
 tf.keras.initializers.RandomNormal(minval=0.0,maxval=0.05,seed=None)
 ```
 3. TruncatedNormal
 Initializer that generates tensors with a truncted normal distribution.
 The values generated are similar to values from a tf.keras.initializers.RandomNormal initializer except that values more than two standard deviations from the mean are discarded and re-drawn.
 ```
 tf.keras.initializers.TruncatedNormal(minval=0.0,maxval=0.05,seed=None)
 ```
 4. Zeros
 Initializer that generates tensors initialized to 0.
 ```
 tf.keras.initializers.Zeros()
 ```
 5. Ones
 Initializer that generates tensors initialized to 1.
 ```
 tf.keras.initializers.Ones()
 ```
 6. GlorotNormal
 
 8. 

Source: [Layer weight initializers](https://keras.io/api/layers/initializers/)
