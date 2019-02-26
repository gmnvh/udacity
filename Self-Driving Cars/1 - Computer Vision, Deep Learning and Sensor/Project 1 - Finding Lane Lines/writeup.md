# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/output_image.png "Algorithm output"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I blurred the image using a size 3 kernel. The blur will help reducing noise in the image.
After that I detected the edges aplying the canny function, restricted the image to a region of interest (ROI) and finally detected the lines using the Hough Transformation.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first filtering all slope that is too different from the one expected as a lane. Then, I classified the line on right lane or left lane. If the slope is negative it is a left lane line. Finally, I used the numpy polyFit function to average the grouped lines into a single one.

Here are some outputs of the algorithm: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when other objetcs (like a car or shadows) get into the ROI. Right now there is no validation if the line founded is actually a lane. Other objects in the ROI could generate false positives.

Another shortcoming could be not detect the lines if the contrast of the road change too much. All the parameters used to detect the lines are very dependeble of the light and road conditions.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a filter to track the lanes between the frames.

Another potential improvement could be to consider curves and find other lanes on the side.
