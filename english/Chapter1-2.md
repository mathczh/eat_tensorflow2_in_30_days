# 1-2 Example: Modeling Procedure for Images


### 1. Data Preparation


The cifar2 dataset is a sub-set of cifar10, which only contains two classes: airplane and automobile.

Each class contains 5000 images for training and 1000 images for testing.

The goal for this task is to train a model to classify images as airplane or automobile.

The files of cifar2 are organized as below:

![](../data/cifar2.jpg)

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

```

<!-- #region -->
There are two ways of image preparation in TensorFlow.

The first one is constructing the image data generator using ImageDataGenerator in tf.keras.

The second one is constructing data pipeline using tf.data.Dataset and several methods in tf.image

The former is simpler and is demonstrated in [this article](https://zhuanlan.zhihu.com/p/67466552) (in Chinese).

```python
train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)
```
1. rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
2. width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
3. shear_range is for randomly applying shearing transformations
4. zoom_range is for randomly zooming inside pictures
5. horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
6. fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

We only apply these for the training data. 

For more detailed explanation, please see this [website](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

The latter is the original method of TensorFlow, which is more flexible with possible better performance with proper usage.

Below is the introduction to the second method.

<!-- #endregion -->

```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
### this is required, otherwise, there is an error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
```

```python

from tensorflow.keras import datasets,layers,models, Sequential, backend
# import tensorflow as tf 
# from tensorflow.compat.v2 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# backend.clear_session()
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
# print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
# tf.config.experimental.set_memory_growth(gpus[0], True)


BATCH_SIZE = 100

def load_image(img_path,size = (32,32)):
    label = tf.constant(1,tf.int8) if tf.strings.regex_full_match(img_path,".*automobile.*") \
            else tf.constant(0,tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img) #In jpeg format
    img = tf.image.resize(img,size)/255.0
    return(img,label)

### use the file name to define the label
### tf.io.read_file returns a tensor of string.
### tf.image.resize() has multiple methods.
```

```python
#Parallel pre-processing using num_parallel_calls and caching data with prefetch function to improve the performance
ds_train = tf.data.Dataset.list_files("../data/cifar2/train/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)  

ds_test = tf.data.Dataset.list_files("../data/cifar2/test/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

### Use the basic functions to handle the raw data.
### tf.data.Dataset.list_files() A dataset of all files matching one or more glob patterns.
### tf.data.experimental.AUTOTUNE will prompt the tf.data runtime to tune the value dynamically at runtime.
### Shuffle(): The dataset fills a buffer with buffer_size elements,\
### then randomly samples elements from this buffer, replacing the selected elements with new elements.\
### For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required. 
### Creates a Dataset that prefetches elements from this dataset. 
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#Checking part of the samples
from matplotlib import pyplot as plt 

plt.figure(figsize=(8,8)) 
for i,(img,label) in enumerate(ds_train.unbatch().take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()

### axesan axes.SubplotBase subclass of Axes (or a subclass of Axes)
### The axes of the subplot. The returned axes base class depends on the projection used.
### It is Axes if rectilinear projection are used and projections.polar.PolarAxes if polar projection are used. 
### The returned axes is then a subplot subclass of the base class.
```

![](../data/1-2-图片预览.jpg)

```python
for x,y in ds_train.take(1):
    print(x.shape,y.shape)
    
### take: Creates a Dataset with at most count elements from this dataset.
```

```
(100, 32, 32, 3) (100,)
```

```python

```

### 2. Model Definition


Usually there are three ways of modeling using APIs of Keras: sequential modeling using `Sequential()` function, arbitrary modeling using functional API, and customized modeling by inheriting base class `Model`.

Here we use API functions for modeling.

```python
tf.keras.backend.clear_session() #Clearing the session

inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32,kernel_size=(3,3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64,kernel_size=(5,5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)
outputs = layers.Dense(1,activation = 'sigmoid')(x)

model = models.Model(inputs = inputs,outputs = outputs)

model.summary()
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        51264     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout (Dropout)            (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 32)                51232     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 103,425
Trainable params: 103,425
Non-trainable params: 0
_________________________________________________________________
```

```python
import tensorflow as tf 

if tf.test.gpu_device_name(): 
    print('Default GPU Device:\
    {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

print(tf.__version__)
```

### 3. Model Training


There are three usual ways for model training: use internal function fit, use internal function train_on_batch, and customized training loop. Here we introduce the simplist way: using internal function fit.

```python
import datetime
# import os

# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = os.path.join('data', 'autograph', stamp)

# We recommend using pathlib under Python3
from pathlib import Path
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('../data/autograph/' + stamp))

print(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"]
    )

```

```python

history = model.fit(ds_train, epochs= 10, validation_data=ds_test,
                    callbacks = [tensorboard_callback], workers = 4, use_multiprocessing=False)

```

```
Train for 100 steps, validate for 20 steps
Epoch 1/10
100/100 [==============================] - 16s 156ms/step - loss: 0.4830 - accuracy: 0.7697 - val_loss: 0.3396 - val_accuracy: 0.8475
Epoch 2/10
100/100 [==============================] - 14s 142ms/step - loss: 0.3437 - accuracy: 0.8469 - val_loss: 0.2997 - val_accuracy: 0.8680
Epoch 3/10
100/100 [==============================] - 13s 131ms/step - loss: 0.2871 - accuracy: 0.8777 - val_loss: 0.2390 - val_accuracy: 0.9015
Epoch 4/10
100/100 [==============================] - 12s 117ms/step - loss: 0.2410 - accuracy: 0.9040 - val_loss: 0.2005 - val_accuracy: 0.9195
Epoch 5/10
100/100 [==============================] - 13s 130ms/step - loss: 0.1992 - accuracy: 0.9213 - val_loss: 0.1949 - val_accuracy: 0.9180
Epoch 6/10
100/100 [==============================] - 14s 136ms/step - loss: 0.1737 - accuracy: 0.9323 - val_loss: 0.1723 - val_accuracy: 0.9275
Epoch 7/10
100/100 [==============================] - 14s 139ms/step - loss: 0.1531 - accuracy: 0.9412 - val_loss: 0.1670 - val_accuracy: 0.9310
Epoch 8/10
100/100 [==============================] - 13s 134ms/step - loss: 0.1299 - accuracy: 0.9525 - val_loss: 0.1553 - val_accuracy: 0.9340
Epoch 9/10
100/100 [==============================] - 14s 137ms/step - loss: 0.1158 - accuracy: 0.9556 - val_loss: 0.1581 - val_accuracy: 0.9340
Epoch 10/10
100/100 [==============================] - 14s 142ms/step - loss: 0.1006 - accuracy: 0.9617 - val_loss: 0.1614 - val_accuracy: 0.9345
```

```python

```

### 4. Model Evaluation

```python
%load_ext tensorboard
#%tensorboard --logdir ../data/keras_model
```

```python
from tensorboard import notebook
notebook.list() 
```

<!-- #raw -->

<!-- #endraw -->

```python
#Checking model in tensorboard
notebook.start("--logdir ../data/autograph")
```

```python

```

![](../data/1-2-tensorboard.jpg)

```python
import pandas as pd 
dfhistory = pd.DataFrame(history.history)
dfhistory.index = range(1,len(dfhistory) + 1)
dfhistory.index.name = 'epoch'

dfhistory
```

![](../data/1-2-dfhistory.jpg)

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
```

```python
plot_metric(history,"loss")
```

![](../data/1-2-Loss曲线.jpg)

```python
plot_metric(history,"accuracy")
```

![](../data/1-2-Accuracy曲线.jpg)

```python
#Evaluating data using model.evaluate function
val_loss,val_accuracy = model.evaluate(ds_test,workers=4)
print(val_loss,val_accuracy)

```

```
0.16139143370091916 0.9345
```


### 5. Model Application


We can use model.predict(ds_test) for prediction.

We can also use model.predict_on_batch(x_test) to predict a batch of data.

```python
model.predict(ds_test)
```

```
array([[9.9996173e-01],
       [9.5104784e-01],
       [2.8648047e-04],
       ...,
       [1.1484033e-03],
       [3.5589080e-02],
       [9.8537153e-01]], dtype=float32)
```

```python
for x,y in ds_test.take(1):
    print(model.predict_on_batch(x[0:20]))
```

```
tf.Tensor(
[[3.8065155e-05]
 [8.8236779e-01]
 [9.1433197e-01]
 [9.9921846e-01]
 [6.4052093e-01]
 [4.9970779e-03]
 [2.6735585e-04]
 [9.9842811e-01]
 [7.9198682e-01]
 [7.4823302e-01]
 [8.7208226e-03]
 [9.3951421e-03]
 [9.9790359e-01]
 [9.9998581e-01]
 [2.1642199e-05]
 [1.7915063e-02]
 [2.5839690e-02]
 [9.7538447e-01]
 [9.7393811e-01]
 [9.7333014e-01]], shape=(20, 1), dtype=float32)
```




```python

```

### 6. Model Saving


We recommend model saving with the original way of TensorFlow.

```python
# Saving the weights, this way only save the tensors of the weights
model.save_weights('../data/tf_model_weights.ckpt',save_format = "tf")
```

```python
# Saving model structure and parameters to a file, so the model allows cross-platform deployment

model.save('../data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('../data/tf_model_savedmodel')
model_loaded.evaluate(ds_test)
```

```
[0.16139124035835267, 0.9345]
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
