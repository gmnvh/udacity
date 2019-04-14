import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

lines = []
with open('./training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Probably because my path is big and have spaces, the images paths
# were divided in two columns. Here I am contanating them.
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

for line in lines:
    filename = line[0].split('\\')[-1]
    current_path = 'G:/data/IMG/' + filename
    images.append(cv2.imread(current_path))
    measurements.append(float(line[3]))

print('Number of training points: ', len(images[0]))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')