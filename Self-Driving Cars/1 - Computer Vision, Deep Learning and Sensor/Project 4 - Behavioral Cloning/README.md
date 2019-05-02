# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

[//]: # (Image References)

[image1]: ./images/center.png "Center"
[image2]: ./images/left.png "Left"
[image3]: ./images/right.png "Right"
[image4]: ./images/hist.png "Hist"


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* `video_with_balanced_data.mp4` (a video recording of your vehicle driving autonomously around the track for at least one full lap using a balanced data set)
* `video_with_imbalanced_data.mp4` (a video recording of your vehicle driving autonomously around the track for at least one full lap using a imbalanced data set)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA architeture recommended during the class lessons and consists of first 2 layers for pre processing the image (lines 104 and 105). These layers are for normalizing and cropping the image.

The hidden layers are composed by 5 convolution layers with RELU activation (lines 108 to 112), one layer to flat the inputs (line 113) and 3 dense layers (lines 114 to 116). Followed by a last dense layer as output (line 119).

```
______________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

While training the model, the number of EPOCHS where adjusted to avoid overfitting. I notice that the model was working fine when trained with 3 epochs but the car was driving out of the track if the model was trained with 5 epochs. I also tried added a dropout layer after the convolutional layers with different dropout rate but the car was driving out of the track for every try.
I dropout layer was also added after the first dense layer but this time I tried increase the number of epochs. But no success on any of the tries.

Finally the model that worked better was the one descriped on item 1 with the following training values:

(With imbalanced data - Using all the data collected)
10814/10814 [==============================] - 36s 3ms/step - loss: 0.0295 - val_loss: 0.0269
Epoch 2/3
10814/10814 [==============================] - 29s 3ms/step - loss: 0.0160 - val_loss: 0.0222
Epoch 3/3
10814/10814 [==============================] - 29s 3ms/step - loss: 0.0137 - val_loss: 0.0245

(With balanced data - Filtering very common steering angles (check item 4 for more details))
5628/5628 [==============================] - 22s 4ms/step - loss: 0.0276 - val_loss: 0.0183
Epoch 2/6
5628/5628 [==============================] - 15s 3ms/step - loss: 0.0170 - val_loss: 0.0179
Epoch 3/6
5628/5628 [==============================] - 16s 3ms/step - loss: 0.0143 - val_loss: 0.0178
Epoch 4/6
5628/5628 [==============================] - 16s 3ms/step - loss: 0.0124 - val_loss: 0.0202
Epoch 5/6
5628/5628 [==============================] - 16s 3ms/step - loss: 0.0103 - val_loss: 0.0206
Epoch 6/6
5628/5628 [==============================] - 16s 3ms/step - loss: 0.0089 - val_loss: 0.0154

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

#### 4. Appropriate training data

Training data was collected driving the car on track 1 for 2 laps. I tried to keep the car in the middle of the track, so that I did not hit the curbs. Some extra data collection was done for sharp turns and on the bridge. The images from the not centered cameras where also used as training data. The steering label used for those images where the steering value from the center image with a correction value of +-0.2 (to left or right). I also flipped the training data images (center, left and right) to get more training data and generalize better the model.
I divided the recorded path in two datasets, one for training and one for validation with a split rate of 0.2 (80% fo the data for training and 20% for validation).

Here is an example of a camera image from the center of the car and the corresponding flipped image:
![alt text][image1]

Here is an example of a camera image from the left of the car and the corresponding flipped image:
![alt text][image2]

Here is an example of a camera image from the right of the car and the corresponding flipped image:
![alt text][image3]

The results for this training data can be found on the video: `video_with_imbalanced_data.mp4`

Looking at the histogram of the steering angles of the training data you can find that the number of samples which steering angle is zero or close to zero is much higher than the rest of the angles. Actually, 3 different groups of angles had a lot of sample (0 - 0.6, -0.32 - -0.18 and 0.18 - 0.23). In order to try a more balanced training set, I reduced the number of samples by 1/3 for each group. The image below shows the histogram before and after the filter.

![alt text][image4]

The results can be found in the video `video_with_balanced_data.mp4`. In my opinion a more balanced data created a smother path, but did not improved much the drive on the bridge. More sample on the bridge would be recommended to improve that.
