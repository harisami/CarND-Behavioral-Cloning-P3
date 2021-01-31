import csv
import cv2
from scipy import ndimage
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout, Lambda, Cropping2D

model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, epochs=3)

model.save('model.h5')