## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/data_visualization.png "Visualization"
[image2]: ./examples/data_hist.png "Histogram"
[image3]: ./examples/gamma.png "Gamma"
[image4]: ./examples/data_augmented.png "Augment"
[image5]: ./examples/augmented_hist.png "Augment Hist"
[image6]: ./examples/gray.png "Gray"
[image7]: ./examples/new_signs.PNG "new signs"
[image8]: ./examples/visualization_weights_1.png" "Weight Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Here are some stastitics about the dataset.

| Summary                     | Data        |
|:---------------------------:|:-----------:|
| Number of training examples | 34799       |
| Number of testing examples  | 12630       |
| Image data shape            | (32, 32, 3) |
| Number of classes           | 43          |


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset.

![alt text][image1]

##### Database Histogram

As part of the exploratory visualization of the dataset, I created the histogram of the training and validation dataset.

It is easy to see that some labels are much more common then other. For example, the database has 2000 ish images of label 2 (Speed limit (50km/h)) and less than 250 images of label 19 (Dangerous curve to the left).

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Gamma Adjustment

Looking at the images in the database I noticed that some of than are very dark. So, as an attempt to improve that, I am pre processing the images doing a gamma adjustment. The code for gamma correction can be found in the `adjust_gamma` function.

Here is an example for image index 18905.

![alt text][image3]

##### Augmenting the training data

In order to augment the training dataset, I am taking all the labels that have less than 750 images and create new images from them. The new images have scale and rotation transformation. The code can be found in the `augment` function.

I am also randomly adding from 300 to 500 images to each selected label.

Here are some samples of augmented signs.

![alt text][image4]

Comparing the histogram now we can see the improvement for labels that were not so frequent before.

![alt text][image5]

##### Convert images to gray scale and normalizing

Since color is not so relevant to the sign classification, removing it from the images will allow the neural network to focus on other more important factors that differentiate the signs.

Here is how the images were converted to gray and normalized.

```python
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)

# Normalize images
X_train_normalized = (X_train_gray - 128)/128 
X_valid_normalized = (X_valid_gray - 128)/128
X_test_normalized = (X_test_gray - 128)/128

```

This is how final processed images look like

![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 gray image                            |
| Convolution           | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Dropout               |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution           | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Dropout               |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | output 400                                    |
| Fully connected       | output 120                                    |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | output 43                                     |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Different combinations of hyper parameters were tested. See below a table with the best combination.

| Parameter             |     Value    |
|:---------------------:|:------------:|
| Gamma                 | 2            |
| Epochs                |100           |
| Batch size            |1000          |
| Mu                    |0             |
| Sigma                 |0.1           |
| Keep probability      |0.8           |
| Learning rate         |0.001         |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I am using the same model from the LetNet Lab lesson with just some dropouts added on each layer. With this configuration I was able to get:

* validation set accuracy of 0.961
* test set accuracy of 0.936
* my 5 new signs set accuracy of 100%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Double Curve          | Double Curve                                  | 
| Speed limit 70Km/h    | Speed limit 70Km/h                            |
| Stop sign             | Stop sign                                     |
| Priority road         | Priority road                                 |
| Go straight or right  | Go straight or right                          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. The high accuracy is probably because the images are very clear with low noise and good light.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

##### Double Curve

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .978                  | Double Curve                                  |
| .012                  | Right-of-way at the next intersection         |
| .007                  | Road work                                     |
| .0                    | Dangerous curve to the left                   |
| .0                    | Turn right ahead                              |

##### Speed limit 70Km/h

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .999                  | Speed limit (70km/h)                          |
| .0                    | Speed limit (30km/h)                          |
| .0                    | Speed limit (20km/h)                          |
| .0                    | Roundabout mandatory                          |
| .0                    | Keep left                                     |

##### Stop sign

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .999                  | Stop                                          |
| .0                    | Turn right ahead                              |
| .0                    | Turn left ahead                               |
| .0                    | Speed limit (60km/h)                          |
| .0                    | No entry                                      |

##### Priority road

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Priority road                                 |
| .0                    | Roundabout mandatory                          |
| .0                    | No passing                                    |
| .0                    | End of all speed and passing limits           |
| .0                    | Right-of-way at the next intersection         |

##### Go straight or right

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .999                  | Go straight or right                          |
| .0                    | End of all speed and passing limits           |
| .0                    | End of no passing                             |
| .0                    | Children crossing                             |
| .0                    | End of speed limit (80km/h)                   |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



The picture below is the visualization for the first weight of the neural network. It seems to be focusing on find the border of the signs, working as some kind of edge detection.

![alt text][image8]


