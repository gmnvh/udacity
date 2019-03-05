



## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/camera_calibration.jpg "Camera calibration"
[image2]: ./output_images/1_undistort_with_input/test1.jpg "Road undistort"
[image3]: ./output_images/2_threshold/test1.jpg "Threshold binary image"
[image4]: ./output_images/3_eagle_eye_poly/straight_lines1.jpg "Source polygon"
[image5]: ./output_images/3_eagle_eye_with_input/straight_lines1.jpg "Eagle eye"
[image6]: ./output_images/4_hist/straight_lines1.jpg "Histogram"
[image7]: ./output_images/5_slidingwindow/straight_lines1.jpg "Polynomial"
[image8]: ./output_images/6_main_output/straight_lines1.jpg "Main output"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `camera_calibration` function.

The goal of this function is to return a camera matrix and distortion coefficients that can be used to undistort the images took from the camera.
The function first creates "object points" of the chessboard corners in the world. Assuming that the chessboard is fixed on the (x,y) plane at z=0, the points will be the same for all the images used during calibration.
Then, it opens each image (best if use 20 to 30 images from multiple angles) and find the chessboard corners using OpenCV function `cv2.findChessboardCorners`. The corners returned by the function is append to a `imgpoints` array.

Finally, with both object points  (`objpoints`) and image corners (`imgpoints`)  it uses `cv2.calibrateCamera()` function to calculate the camera matrix and distortion coefficients.

The result can be seen in the image below:

![Camera calibration image][image1]

### Pipeline (single images)

> NOTE: All the code for this project is in a single file called
> `main.py`. Even knowing that it is not ideal to put everything into a
> single file. I believe that it makes it easier to review and share.

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the `Lane.undistort` function of the `main.py`.

After initializing the class `Lane` with the camera matrix and distortion coefficients, the function `Lane.undistort` can be called to undistort any image. The implementation is simply a call to the OpenCV function `cv2.undistort`.

```python
def undistort(self, image):
    """
    This function will use the camera calibration outpus to undistort the image.
    
    img - BGR image
    """
    return cv2.undistort(image, self.camera_mtx, self.camera_dist, None, self.camera_mtx)
```

Here is an example of an undistort image:

![Road undistort][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function `Lane.thresh_pipeline`).   The color threshold was done first converting the image to HSL and then applying a threshold in the saturation channel.

For the gradient threshold multiple steps were taken:
* Take the derivative on x using `cv2.Sobel` and then applying a threshold;
* Take also the derivate on y and calculate the magnitude and direction of the gradient vector;
* Apply a threshold for a range of specific magnitudes and directions;
* Combine all thresholds.

The combination is shown below, where `s_binary` is the binary image generated by the color threshold, `mag_binary` is the magnitude one, `dir_binary` is the direction and `sx_binary` the gradient in the x direction.

```python
combined_binary = np.zeros_like(dir_binary)
combined_binary[(sx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
```

Here's an example of my output for this step:
![Threshold binary image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `Lane.warper`. The function takes as input an image (`img`), the source (`src`) and destination (`dst`) points are hardcoded in the `Lane.__init__` constructor:

```python
self.poly_src = np.float32([[190,720], [1130, 720], [705, 455], [585, 455]])

offset = 300
img_size = (1280, 720)
self.poly_dst = np.float32([[offset,img_size[1]], [img_size[0]-offset, 720],
                            [img_size[0]-offset, 0], [offset, 0]])
```

The source polygon can be seeing in the image below:

![Source polygon][image4]

And below I have the perspective transformation in the binary image:

![Eagle eye][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The first thing to start extracting the points that are part of each lane line is to localize where the lines start. This is done in function `Lane.find_lane_base`. This function will calculate the histogram along the x axis and find the 2 maximum points from the center of the image. The image below shows an example of that but you can also find a very cool video on how that works on this [link to histogram video](./output_videos/4_hist/project_video.mp4).

![Histogram][image6]

After that, a sliding window method is applied to locate each line points. The implementation for the sliding window method can be found in `Lane.find_lane_pixels`. And it will return all the (x, y) coordinates for pixel that are part of the lane line. The  `Lane.find_lane_pixels` is actually called by the function `Lane.polynomial` that fits a polynomial for each lane line pixels found. Here is an example:

![Polynomial][image7]

All the polynomial information is saved in `Lane` class variables to be used later.

```python
# save fit to class
self.size = (binary_warped.shape)
self.left_fit = left_fit
self.right_fit = right_fit
self.left_fitx = left_fitx
self.right_fitx = right_fitx
self.left_fit_cr = left_fit_cr   # real world fit
self.right_fit_cr = right_fit_cr # real world fit
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated using the formula below, where A and B are the fitted line coefficients. The function `Lane.lane_curvature` is used to update the `Lane` instance with both left and right curvature.

```python
def curvature(self, A, B, y_eval):
    aux = (2*A*y_eval + B)**2
    aux = (1 + aux)**(1.5)
    aux = aux / np.absolute(2 * A)
    return aux
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After all the calculations, the result is printed on the original image using the functions `Lane.print_curvature` and `Lane.print_lane`. The first one will populate the text information and the second one will print the green lane. I am also printing a gray line as the center of the lane.

![output image][image8]

---
### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For videos, I implement a function called 'Lane.sanity_check` that will drop any frame where the calculated lane lines do not meet the establish criteria.

The function checks for:
* left and right curvature difference;
* distance between left and right lines at the top and at the bottom.

Definitly, there is a lot of room for improvement here. The idea is to come back after finishing the Udacity NanoDegree and make the code more robust.

The video pipeline also uses the search from prior algorithm taught on lesson 9.5 in case a lane was detected before. I did not like much the results of it. It seems to be more sensitive to noise. So, a improvement to `sanity_check` function could make better use of the technic.

Here's a [link to my video result](./output_videos/6_main_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The challenge videos have shown that there are much more to be learned about lane detection. The pipeline is very sensitive to any noise or other lines on the road. Probably because the main way to separate the line from the image is the threshold part. Also, very sharp curves are a big challenge and it makes this algothrim fails.

The 'sanity_check' function is another part that could be revisited to improve the performance and, of course, spent a lot of more hours tunning the threshold pipeline.
