import os
import csv
import cv2
from scipy import ndimage
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math

### reading in the driving_log csv file
### and collecting each row detail individually
samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

### splitting the data
from sklearn.model_selection import train_test_split
training_samples, validation_samples = train_test_split(samples, test_size=0.25)

### generator function definition
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # reading in images from center, left and right cameras
                img_center = ndimage.imread('../data/IMG/' + batch_sample[0].split('/')[-1])
                img_left = ndimage.imread('../data/IMG/' + batch_sample[1].split('/')[-1])
                img_right = ndimage.imread('../data/IMG/' + batch_sample[2].split('/')[-1])
                
                images.extend((img_center, img_left, img_right))
                measurements.extend((steering_center, steering_left, steering_right))

            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# calling the generator function on training and validation samples
training_generator = generator(training_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D

model = Sequential()

# preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
# convolutional layers
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

# configuring the model with mean squared error loss and adam optimizer for training
model.compile(loss='mse', optimizer='adam')
# training the model for the said epochs
history_object = model.fit_generator(training_generator,
                                    steps_per_epoch=math.ceil(len(training_samples)/batch_size),
                                    validation_data=validation_generator,
                                    validation_steps=math.ceil(len(validation_samples)/batch_size),
                                    epochs=3, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch and saving the image plot
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model - Mean Squared Error Loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./examples/visualization.jpg')

### saving a cropped version of a sample training image
img = ndimage.imread('./examples/sample_image.jpg')
img_cropped = img[70:135, :, :]
print(img_cropped.shape)
scipy.misc.imsave('./examples/sample_image_cropped.jpg', img_cropped)