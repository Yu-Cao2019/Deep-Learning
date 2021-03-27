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
 Draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units in the weight tensor.
 ```
 tf.keras.initializers.GlorotNormal(seed=None)
 ```
 7. Glorotuniform
 Draws samples from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units).
 ```
 tf.keras.initializers.GlorotUniform(seed=None)
 ```
 8. Identity
 Initializer that generates the identity matrix.
 ```
 tf.keras.initializers.Identity(gain=1.0)
 ```
 where `gain` means the multiplicative factor to apply to the identity matrix.
 9. Orthogonal
 Initializer that generates the orthogonal matrix.
 ```
 tf.keras.initializers.orthogonal(gain=1.0, seed=None)
 ```
 10. Constant
 Initializer that generates with constant values.
 ```
 tf.keras.initializers.Constant(value=0)
 ```
 11. VarianceScaling
 Initializer capable of adapting its scale to the shape of weights tensors.
 ```
 tf.keras.initializers.VarianceScaling(scale=1.0,mode='gan_in',distribution='truncated_normal',seed=None)
 ```
 


Source: [Layer weight initializers](https://keras.io/api/layers/initializers/)
