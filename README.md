**# -Cat-Vs-Dog-Image-Classification**
** Cat Vs Dog Image Classification using CNN model**

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile

zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')

zip_ref.extractall('/content')

zip_ref.close()

import tensorflow as tf

from tensorflow import keras

from keras import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout


**#generators**

train_ds = keras.utils.image_dataset_from_directory(

    directory = '/content/train',
				
    labels='inferred',
				
    label_mode = 'int',
				
    batch_size=32,
				
    image_size=(256,256)
)


validation_ds = keras.utils.image_dataset_from_directory(

    directory = '/content/test',
				
    labels='inferred',
				
    label_mode = 'int',
				
    batch_size=32,
				
    image_size=(256,256)
				
)

**# Normalize**

def process(image,label):

    image = tf.cast(image/255. ,tf.float32)
				
    return image,label

train_ds = train_ds.map(process)

validation_ds = validation_ds.map(process)

**# create CNN model**

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')

plt.plot(history.history['val_accuracy'],color='blue',label='validation')

plt.legend()

plt.show()

![image](https://github.com/NehaMore2202/-Cat-Vs-Dog-Image-Classification/assets/154467395/5738bb83-2503-4d74-a476-1d432986e3b9)

plt.plot(history.history['accuracy'],color='red',label='train')

plt.plot(history.history['val_accuracy'],color='blue',label='validation')

plt.legend()

plt.show()

![image](https://github.com/NehaMore2202/-Cat-Vs-Dog-Image-Classification/assets/154467395/e751d047-ff68-4cc1-a0a9-598ce17caed7)

plt.plot(history.history['loss'],color='red',label='train')

plt.plot(history.history['val_loss'],color='blue',label='validation')

plt.legend()

plt.show()

![image](https://github.com/NehaMore2202/-Cat-Vs-Dog-Image-Classification/assets/154467395/c0a86112-32cd-40e7-8117-fc32629673d8)



