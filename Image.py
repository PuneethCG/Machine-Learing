import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation="sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('E:\AI-DL\CNN\parent',target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('E:\AI-DL\CNN\parent',target_size=(64,64),batch_size=32,class_mode='binary')

clr=model.fit_generator(train_set,steps_per_epoch=800,epochs=1,validation_data=test_set,validation_steps=200)

model.save('model.h5')
print("Saved model to disk")

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
test_image =image.load_img(r'E:\AI-DL\CNN\parent\tiger\test.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
model = load_model('model.h5')
result=model.predict(test_image)
train_set.class_indices
if result[0][0]==1:
    predection = "tiger"
    print(predection)
else:
    predection = 'lion'
    print(predection)
