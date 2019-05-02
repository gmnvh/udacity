import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Probably because my path is big and have spaces, the images paths
# were divided in two columns. Here I am concatanating them.
if len(lines[0]) == 10:
    new_lines = []
    for line in lines:
        # create new lines adding the path columns
        new_line = []
        new_line.append(line[0]+line[1])
        new_line.append(line[2]+line[3])
        new_line.append(line[4]+line[5])
        new_line.extend(line[6:10])
        
        # append to the list of lines
        new_lines.append(new_line)
    
    # replace lines with the new formatted lines
    lines = new_lines

# Each line should have 7 columns
assert len(lines[0]) == 7, 'Each line should have 7 columns'

# Open images
images = []
measurements = []

# Steering correction factor for images not in the center of the car
correction = [0, 0.2, -0.2]
for line in lines:
    for i in range(3):
        filename = line[i].split('\\')[-1]
        current_path = './data/IMG/' + filename
        images.append(cv2.imread(current_path))
        measurements.append(float(line[3]) + (correction[i]))

# Augmented images - create new images flipping the existing
# ones and inverting the steering input
new_images = []
new_measurements = []
for image, measurement in zip(images, measurements):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    new_images.append(image_flipped)
    new_measurements.append(measurement_flipped)

# Add the new flipped images to the list
images.extend(new_images)
measurements.extend(new_measurements)	
print('Number of training points: ', len(images))

X_train = np.array(images)
y_train = np.array(measurements)
y_train_before = y_train

from numpy.random import permutation
perm = permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

around_zero = ((y_train > -0.01) & (y_train < 0.06)) | ((y_train > -0.32) & (y_train < -0.18)) | ((y_train > 0.18) & (y_train < 0.23))

y_train_1 = y_train[np.where(around_zero == False)]
X_train_1 = X_train[np.where(around_zero == False)]
y_train_2 = y_train[np.where(around_zero != False)]
X_train_2 = X_train[np.where(around_zero != False)]

y_train_3 = y_train_2[0:int(len(y_train_2)/3)]
X_train_3 = X_train_2[0:int(len(y_train_2)/3)]
y_train_4 = np.concatenate((y_train_1, y_train_3), axis=None)

X_train_4 = np.concatenate((X_train_1, X_train_3), axis=0)

X_train = X_train_4
y_train = y_train_4

print(len(X_train), len(y_train))
assert(len(X_train) == len(y_train))


import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(y_train_before, bins=40)
axs[1].hist(y_train, bins=40)

plt.show()

# NN model
model = Sequential()

# Image Pre-Processing
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

# Hidden layers
model.add(Convolution2D(24,5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3, 3, activation="relu"))
model.add(Convolution2D(64,3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 

# Output layer
model.add(Dense(1))

# Compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=6)

# Save trainned model
model.save('model.h5')

print(model.summary())