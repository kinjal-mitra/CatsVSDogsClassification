#Run the code after downloading the zip file and storing in the same folder

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# 1.Importing Necessary Libraries

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cv2

#Generators divide the datset into smaller batches and the load each batch at a time to the ram for processing. This is a very useful attribute of keras, which is essential for handling large dataset.

#Creating Generators for Data Loading


#Generator for our images dataset.
train_df = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int', #This assigns 0 to cats and 1 to dogs
    batch_size = 64,
    image_size = (256,256)
)

test_df = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels = 'inferred',
    label_mode = 'int', #This assigns 0 to cats and 1 to dogs
    batch_size = 64,
    image_size = (256,256)
)

#This stores the data in the form of numpy array with values ranging from 0-255. However, we need it in the range of 0-1. Hence, we normalize the same.


# Normalization
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label

train_df = train_df.map(process)
test_df = test_df.map(process)

"""CNN Model

This CNN Model consists for 3 convolutional layers.
First Layer consists of 32 filters
Second Layer - 64 filters
Third Layer - 128 filters
"""

model = Sequential()

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

#model.summary()

# Optimizer
optimizer = Adam(learning_rate=0.001)

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compile Model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_df, epochs=100, validation_data=test_df,
                    callbacks=[lr_scheduler, early_stopping])



"""Saving the Model"""

tf.saved_model.save(model,".")