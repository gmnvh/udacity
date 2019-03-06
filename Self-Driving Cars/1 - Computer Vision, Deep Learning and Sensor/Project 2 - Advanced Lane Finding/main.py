import os
import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from matplotlib.pyplot import gray

# Camera calibration
#----------------------------

# Project camera
projcam_calibration_images = r'./camera_cal'
projcam_calibration_test_image = r'./camera_cal/calibration1.jpg'
projcam_chessboard_corner_x = 9
projcam_chessboard_corner_y = 6
projcam_out_camera_calibration = r'./examples/camera_calibration.jpg'

# My camera
mycam_calibration_images = r'./mycamera_cal'
mycam_calibration_test_image = r'./mycamera_cal/calibration14.jpg'
mycam_chessboard_corner_x = 9
mycam_chessboard_corner_y = 6
mycam_out_camera_calibration = r'./examples/mycamera_calibration.jpg'

# Select camera used
calibration_images = projcam_calibration_images
calibration_test_image = projcam_calibration_test_image
chessboard_corner_x = projcam_chessboard_corner_x
chessboard_corner_y = projcam_chessboard_corner_y
out_camera_calibration = projcam_out_camera_calibration

# Undistortion example
#----------------------------
image_to_undistort = r'test_images/test5.jpg'
out_image_to_undistort = r'./examples/test5_undistort.jpg'

#----------------------------------------------------------------------------------------------------
# Camera calibration
#----------------------------------------------------------------------------------------------------

def camera_calibration():
    print('Performing camera calibration')
    
    # read calibration files
    cal_files = glob.glob((calibration_images + r'/calibration*.jpg'))
    print('Found', len(cal_files), ' calibration files')
    assert len(cal_files) >= 20, 'A minimum of 20 calibration files are recomended'

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_corner_y * chessboard_corner_x, 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_corner_x, 0:chessboard_corner_y].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for cal_file in cal_files:
        # open gray image
        img = cv2.imread(cal_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_corner_x, chessboard_corner_y), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    print('Camera calibration completed')
    return mtx, dist


def show_camera_calibration(mtx, dist, save_to_file=False):
    # open test image and undistort
    img = cv2.imread(calibration_test_image)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    # visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    
    if save_to_file:
        plt.savefig(out_camera_calibration)
        print('Camera calibration example saved to file', out_camera_calibration)
    else:
        plt.show()


def show_undistortion(img_file, mtx, dist, save_to_file=False):
    # open test image and undistort
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    # visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    
    if save_to_file:
        plt.savefig(out_image_to_undistort)
        print('Undistort road example saved to file', out_image_to_undistort)
    else:
        plt.show()


#----------------------------------------------------------------------------------------------------
# Image and video display
#----------------------------------------------------------------------------------------------------

def img_1C_to_3C(img):
    mult = 1
    if np.amax(img) == 1: # check if it is a binary image
        mult = 255
    if len(img.shape) == 2: # check single channel frame
            img = np.dstack((img, img, img)) * mult
    return img

def save_images(img_dir, out_dir, process_func=None, show=True, show_input=False, resize=0.7, reset=None):
    image_list = os.listdir(img_dir)
    
    for image_file in image_list:
        img = cv2.imread(img_dir + image_file, cv2.IMREAD_COLOR)
        img_input = img.copy()

        # process image if callback function is defined
        output_file = out_dir + image_file
        if process_func is not None:
            img = process_func(img)
            img = img_1C_to_3C(img)

        # show input image in the left and process in the right
        if show_input is True:
            img = np.concatenate((img_input, img), axis=1)
            
        # write image to file
        cv2.imwrite(output_file, img)
        print('File saved:', output_file)

        # show image in a window
        if show is True:
            cv2.namedWindow(output_file)
            cv2.moveWindow(output_file, 0, 0)
            img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
            cv2.imshow(output_file, img)
            k = cv2.waitKey()
            cv2.destroyWindow(output_file)
            if chr(k) == 'q':
                break
        
        # reset Lane
        if reset is not None:
            reset()

def save_videos(video_dir, out_dir, process_func=None, show=True, show_input=False, resize=0.7, save_frame=[], reset=None):
    video_list = os.listdir(video_dir)

    for video_file in video_list:

        # get input video properties
        cap = cv2.VideoCapture(video_dir + video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('Video input:', video_file, int(fps), 'fps ', size)

        # create output video handler
        output_file = out_dir + video_file
        #fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        if show_input is True:
            size = (2*size[0], size[1])
            print('Resizing video for', size)
        out = cv2.VideoWriter(output_file, fourcc, fps, size)

        if show is True:
            cv2.namedWindow(output_file)
            cv2.moveWindow(output_file, 0, 0)

        # process video
        ret, frame = cap.read()
        frame_counter = 0
        
        while ret is True:
            start_time = time.time()
            frame_input = frame.copy()

            # process frame
            if process_func is not None:
                frame = process_func(frame)
                frame = img_1C_to_3C(frame)

            # show input image in the left and process in the right
            if show_input is True:
                frame = np.concatenate((frame_input, frame), axis=1)
                
            # save output video
            out.write(frame)

            # show video
            if show is True:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
                cv2.imshow(output_file, frame)

                elapsed_time = time.time() - start_time
                #cfps = max(0, int(fps - elapsed_time * 100)) - 10
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_counter in save_frame:
                img_file = out_dir + str(frame_counter) + '.jpg'
                cv2.imwrite(img_file + '', frame)
                
            # read next frame
            ret, frame = cap.read()
            frame_counter += 1

        out.release()
        cv2.destroyWindow(output_file)
        if reset is not None:
            reset()

    cap.release()
    cv2.destroyAllWindows()
    return

#----------------------------------------------------------------------------------------------------
# Thresholded Binary Image
#----------------------------------------------------------------------------------------------------

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply the following steps to img
  
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply the following steps to img  
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # 3) Calculate the magnitude
    magn = cv2.magnitude(sobelx, sobely)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magn/np.max(magn))
    
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply the following steps to img
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gray)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    
    # Return this mask as your binary_output image
    return binary_output

def hsl_thresh(img, s_thresh=(170, 255)):
    # 1) Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    
    # 2) Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # 3) Return this mask as your binary_output image
    return s_binary

def hsl_grad_thresh(img, s_thresh=(30, 160)):
    # 1) Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    #s_channel = cv2.equalizeHist(s_channel)
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    scaled = np.uint8(255.0*abs_sobel/np.max(abs_sobel))
    
    x_binary = np.zeros_like(scaled)
    x_binary[(scaled >= s_thresh[0]) & (scaled <= s_thresh[1])] = 1
    
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobely)
    scaled = np.uint8(255.0*abs_sobel/np.max(abs_sobel))
    
    y_binary = np.zeros_like(scaled)
    y_binary[(scaled >= s_thresh[0]) & (scaled <= s_thresh[1])] = 1
    
     # Threshold color channel
    s_binary = np.zeros_like(x_binary)
    s_binary[(x_binary >= 1) & (y_binary <= 1)] = 1
    
    # Return this mask as your binary_output image
    return s_binary


#----------------------------------------------------------------------------------------------------
# Line class
#----------------------------------------------------------------------------------------------------

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


#----------------------------------------------------------------------------------------------------
# Lane class
#----------------------------------------------------------------------------------------------------

class Lane():
    # camera calibration
    camera_mtx = None
    camera_dist = None

    # perpective transformation
    poly_src = None
    poly_dst = None
    poly_han = None

    # debug auxiliar variables
    frame_counter = 0
    dropped_frames = 0
    total_dropped_frames = 0
    error_left_curv = []

    # average for lane curvature
    left_avg_curv = 0
    right_avg_curv = 0

    # display options
    # if True, the display functions for the variables below will consider the input
    # as an original file (BGR image)
    from_original = False # thresh_pipeline_display


    def __init__(self):
        self.average_size = 15
        self.left_line = Line()
        self.right_line = Line()
        self.poly_src = np.float32([[190,720], [1130, 720], [705, 455], [585, 455]])

        offset = 300
        img_size = (1280, 720)
        self.poly_dst = np.float32([[offset,img_size[1]], [img_size[0]-offset, 720],
                                    [img_size[0]-offset, 0], [offset, 0]])


    def reset(self):
        print("Dropped:", self.total_dropped_frames)
        self.left_avg_curv = 0
        self.right_avg_curv = 0
        
        self.frame_counter = 0
        self.dropped_frames = 0
        self.total_dropped_frames = 0
        self.error_left_curv = []
        
        self.left_line = Line()
        self.right_line = Line()

    def undistort(self, image):
        '''
        This function will use the camera calibration outpus to undistort the image.

        img - BGR image
        '''
        return cv2.undistort(image, self.camera_mtx, self.camera_dist, None, self.camera_mtx)


    def thresh_pipeline(self, img):
        '''
        Threshold pipeline to return a binary image (only 1 channel). This function must be used
        by the code pipeline but can not be used for display. The save_videos function needs a 
        3 channel image as a return. For that use `thresh_pipeline_display` function.

        img - BGR image undistorted
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        s_binary = hsl_thresh(img, s_thresh=(130, 255))
        sx_binary = abs_sobel_thresh(gray, 'x', 3, (30, 100))
        mag_binary = mag_thresh(gray, mag_thresh=(100, 255))
        dir_binary = dir_threshold(gray, thresh=(0.8, 1.2))

        grad_binary = np.zeros_like(dir_binary)
        grad_binary[(sx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        combined_binary = np.zeros_like(dir_binary)
        combined_binary[(sx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

        return combined_binary, s_binary, grad_binary


    def warp(self, img):
        '''
        Warp the image to create the eagle eye view.

        img - the function can warp any type of input image
        '''
        if self.poly_han is None:
            self.poly_han = cv2.getPerspectiveTransform(self.poly_src, self.poly_dst)
            print("Perspective calculated")

        warped = cv2.warpPerspective(img, self.poly_han, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped


    def hist(self, img):
        '''
        Calculate the histogram of bottom half of an image.

        img - 1 channel image
        '''
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0]//2:,:]

        # Sum across image pixels vertically - make sure to set `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)
        
        return histogram


    def find_lane_base(self, hist):
        '''
        This function will return the left and right line base position.
        
        hist - image histogram
        '''
        midpoint = np.int(hist.shape[0]//2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint
        return leftx_base, rightx_base


    def find_lane_pixels(self, binary_warped):
        '''
        Extract all the points for each lane line.
        
        binary_warped - Binary warped image
        '''
        # Take a histogram of the bottom half of the image
        histogram = self.hist(binary_warped)
    
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base, rightx_base = self.find_lane_base(histogram)

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        prev_x = int(left_fitx[0])
        prev_y = int(ploty[0])
        for x, y in zip(left_fitx[1:], ploty[1:]):
            cv2.line(out_img, (prev_x, prev_y), (int(x), int(y)), (255, 255, 255), 10)
            prev_x = int(x)
            prev_y = int(y)

        prev_x = int(right_fitx[0])
        prev_y = int(ploty[0])
        for x, y in zip(right_fitx[1:], ploty[1:]):
            cv2.line(out_img, (prev_x, prev_y), (int(x), int(y)), (255, 255, 255), 10)
            prev_x = int(x)
            prev_y = int(y)

        # save fit to class
        self.size = (binary_warped.shape)
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.left_fit_cr = left_fit_cr   # real world fit
        self.right_fit_cr = right_fit_cr # real world fit

        return out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit()
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = np.poly1d(left_fit)(ploty)
        right_fitx = np.poly1d(right_fit)(ploty)
        
        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100
    
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        l = np.poly1d(self.left_fit)
        r = np.poly1d(self.right_fit)
        left_lane_inds = ((nonzerox >= l(nonzeroy)-margin) & 
                          (nonzerox < l(nonzeroy)+margin))
        right_lane_inds = ((nonzerox >= r(nonzeroy)-margin) & 
                          (nonzerox < r(nonzeroy)+margin))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        left_fit_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        
        # save fit to class
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.left_fit_cr = left_fit_cr
        self.right_fit_cr = right_fit_cr
        
        # lane center
        self.lane_center = ((right_fitx[len(ploty)-1] - left_fitx[len(ploty)-1])/2) + left_fitx[len(ploty)-1]
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
        
        return result

    def curvature(self, A, B, y_eval):
        aux = (2*A*y_eval + B)**2
        aux = (1 + aux)**(1.5)
        aux = aux / np.absolute(2 * A)
        return aux

    def lane_curvature(self):
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image   
        y_eval = self.size[0] * self.ym_per_pix
        
        self.left_curv = self.curvature(self.left_fit_cr[0], self.left_fit_cr[1], y_eval)
        self.right_curv = self.curvature(self.right_fit_cr[0], self.right_fit_cr[1], y_eval)

    def print_lane(self, img):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        # Create an image to draw the lines on
        warp_zero = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if self.left_line.bestx is not None:
            left_fitx = self.left_line.bestx
            right_fitx = self.right_line.bestx

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

            # Draw the center of the lane
            prev_leftx = left_fitx[0]
            prev_rightx = right_fitx[0]
            prev_y = ploty[0]
            for lx, rx, y in zip(left_fitx[1:], 
                                 right_fitx[1:], ploty[1:]):
                cv2.line(color_warp, (int(((prev_rightx-prev_leftx)//2)+prev_leftx), int(prev_y)), (int(((rx-lx)//2)+lx), int(y)), (255, 255, 255), 10)
                prev_leftx = lx
                prev_rightx = rx
                prev_y = int(y)

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            Minv = cv2.getPerspectiveTransform(self.poly_dst, self.poly_src)
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

            # Combine the result with the original image
            img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return img

    def print_lane_curvature(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX

        n = self.average_size
        if self.left_avg_curv == 0: # initial state
            self.left_avg_curv = self.left_curv
            self.right_avg_curv = self.right_curv

        self.left_avg_curv = (((n-1)*self.left_avg_curv) + self.left_curv)/n
        self.right_avg_curv = (((n-1)*self.right_avg_curv) + self.right_curv)/n

        if self.left_line.bestx is not None:
            lane_curvature = (self.left_avg_curv + self.right_avg_curv)/2
            cv2.putText(img,"Lane curvature %d m"%lane_curvature,(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)

             # lane center
            self.lane_center = ((self.right_line.bestx[self.size[0]-1] - self.left_line.bestx[self.size[0]-1])/2) + self.left_line.bestx[self.size[0]-1]

            dist_from_center = self.lane_center - self.size[1]//2
            self.dist_from_center = dist_from_center * self.xm_per_pix

            if self.dist_from_center > 0:
                message = '{:.2f} m right of center'.format(self.dist_from_center)
            else:
                message = '{:.2f} m left of center'.format(-self.dist_from_center)

            cv2.putText(img, message,(10,80), font, 1,(255,255,255),2,cv2.LINE_AA)
        return img

    def sanity_check(self):
        drop_frame = False

        # Curvature
        diff_curv = self.left_curv - self.right_curv
        self.error_left_curv.append(diff_curv)
            
        # anything curvature greater than 3000 will be consider a straith line and
        # very sensitive to noise
        if self.left_curv < 3000 or self.right_curv < 3000:
            if np.absolute(diff_curv) > 2000:
                drop_frame = True
                self.dropped_frames += 1
                self.total_dropped_frames += 1
                print("Frame", self.frame_counter, "with curvature diff of", diff_curv)

        # drop frame if distance from top is too different from bottom
        diff_distance_top = self.right_fitx[0] - self.left_fitx[0]
        diff_distance_bottom = self.right_fitx[self.size[0]-1] - self.left_fitx[self.size[0]-1]

        diff = abs(diff_distance_bottom - diff_distance_top)
        if drop_frame is False and (diff_distance_top < 0 or diff > 200):
            drop_frame = True
            self.dropped_frames += 1
            self.total_dropped_frames += 1
            print("Frame", self.frame_counter, "with distance of", diff)

        if self.dropped_frames >= 4:
            self.left_line.detected = False
            self.right_line.detected = False

        # if candidate lane is ok
        if drop_frame is False:
            if self.dropped_frames != 0:
                print("Dropped frame before finding lane: ", self.dropped_frames)

            self.left_line.detected = True
            self.right_line.detected = True

            self.dropped_frames = 0
            self.left_line.recent_xfitted.insert(0, self.left_fitx)
            self.left_line.recent_xfitted = self.left_line.recent_xfitted[:self.average_size]

            self.right_line.recent_xfitted.insert(0, self.right_fitx)
            self.right_line.recent_xfitted = self.right_line.recent_xfitted[:self.average_size]

            self.left_line.bestx = np.average(self.left_line.recent_xfitted, axis=0)
            self.right_line.bestx = np.average(self.right_line.recent_xfitted, axis=0)

            self.left_line.radius_of_curvature = self.left_curv
            self.right_line.radius_of_curvature = self.right_curv

        self.frame_counter += 1

    def main_pipeline(self, img):
        img_p = self.undistort(img)
        img_p, _, _ = self.thresh_pipeline(img_p)
        img_p = self.warp(img_p)
        
        if self.left_line.detected is False:
            img_out = self.fit_polynomial(img_p)
        else:
            img_out = self.search_around_poly(img_p)

        self.lane_curvature()
        self.sanity_check()
        img = self.print_lane_curvature(img)
        return self.print_lane(img)


    #----------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------
    # Display helper functions
    #----------------------------------------------------------------------------------------------------
    def thresh_pipeline_display(self, img):
        '''
        Threshold pipeline to return a binary image with 3 channels. This function must be used
        only to generate display images or videos of the threshold process. Must not be used by
        the main lane detection pipeline. For that use `thresh_pipeline` function.
        
        img - BGR undistorted image or, if from_original is True, BGR original image
        '''

        if self.from_original is True:
            img = self.undistort(img)

        combined, s_binary, grad_binary = self.thresh_pipeline(img)

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((s_binary, grad_binary, grad_binary)) * 255

        return color_binary

    def perspective_display(self, img):
        roi = Image.fromarray(img)
        d = ImageDraw.Draw(roi)
        d.polygon(self.poly_src, fill=None, outline=(20, 20, 255))
        return np.asarray(roi)

    def warp_display(self, img):
        if self.from_original is True:
            img = self.undistort(img)
            img, _, _ = self.thresh_pipeline(img)
        return self.warp(img)

    def hist_display(self, img):
        '''
        This function will display the histogram of a warped image.
        '''
        if self.from_original is True:
            img = self.undistort(img)
            img, _, _ = self.thresh_pipeline(img)
            img = self.warp(img)

        hist = self.hist(img)

        # draw histogram to a canvas
        fig = plt.figure(figsize=(12.8,7.2))
        xs = np.arange(img.shape[0])
        plt.plot(hist)
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        plt.close(fig)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        return img

    def fit_polynomial_display(self, img):
        if self.from_original is True:
            img = self.undistort(img)
            img, _, _ = self.thresh_pipeline(img)
            img = self.warp(img)
        else:
            # if image is not from original assume it is a
            # warped binary eagle view image
            #roi  = Image.fromarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        out_img = self.fit_polynomial(img)
        return out_img


if __name__ == '__main__':
    #######
    # Local use only
    #######
    #import os
    os.chdir(r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 2 - Advanced Lane Finding')
    #os.chdir(r'C:\Gustavo\projects\udacity\Self-Driving Cars\github\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 2 - Advanced Lane Finding')
    #print(os.path.abspath("./"))
    #######
    
    mtx, dist = camera_calibration()
    #show_camera_calibration(mtx, dist, True)

    # create lane object and initialize camera parameters
    lane = Lane()
    lane.camera_mtx = mtx
    lane.camera_dist = dist

    # undistort images
    input = r"./test_images/"
    output =  r'./output_images/1_undistort/'
    #save_images(input, output, process_func=lane.undistort, show_input=False, show=False, reset=lane.reset)
    output =  r'./output_images/1_undistort_with_input/'
    #save_images(input, output, process_func=lane.undistort, show_input=True, show=False, reset=lane.reset)
    input = r"./test_videos/"
    output =  r'./output_videos/1_undistort/'
    #save_videos(input, output, process_func=lane.undistort, show_input=False, show=False, reset=lane.reset)

    # threshold - colored binary for display
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/2_threshold/'
    #save_images(input, output, process_func=lane.thresh_pipeline_display, show_input=True, show=False, reset=lane.reset)
    input = r'./output_videos/1_undistort/'
    output =  r'./output_videos/2_threshold/'
    #save_videos(input, output, process_func=lane.thresh_pipeline_display, show_input=True, show=False, reset=lane.reset)

    # perspective display
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/3_eagle_eye_poly/'
    #save_images(input, output, process_func=lane.perspective_display, show_input=True, reset=lane.reset)

    # eagle's eye
    lane.from_original = True
    input = r"./test_images/"
    output =  r'./output_images/3_eagle_eye/'
    #save_images(input, output, process_func=lane.warp, show_input=False, show=False, reset=lane.reset)
    output =  r'./output_images/3_eagle_eye_with_input/'
    #save_images(input, output, process_func=lane.warp, show_input=True, show=False, reset=lane.reset)
    input = r"./test_videos/"
    output =  r'./output_videos/3_eagle_eye/'
    #save_videos(input, output, process_func=lane.warp_display, show_input=False, show=False, reset=lane.reset)
    output =  r'./output_videos/3_eagle_eye_with_input/'
    #save_videos(input, output, process_func=lane.warp_display, show_input=True, show=False, reset=lane.reset)
    lane.from_original = False

    # histogram
    input = r'./output_images/3_eagle_eye/'
    output =  r'./output_images/4_hist/'
    #save_images(input, output, process_func=lane.hist_display, show_input=True, show=False, reset=lane.reset)
    input = r'./output_videos/3_eagle_eye/'
    output =  r'./output_videos/4_hist/'
    #save_videos(input, output, process_func=lane.hist_display, show_input=True, show=False, reset=lane.reset)

    # Define conversions in x and y from pixels space to meters
    lane.ym_per_pix = 30/720 # meters per pixel in y dimension
    lane.xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # sliding window
    lane.from_original = True
    input = r"./test_images/"
    output =  r'./output_images/5_slidingwindow/'
    #save_images(input, output, process_func=lane.fit_polynomial_display, show_input=True, show=False, reset=lane.reset)
    input = r"./test_videos/"
    output =  r'./output_videos/5_slidingwindow/'
    #save_videos(input, output, process_func=lane.fit_polynomial_display, show_input=True, show=False, reset=lane.reset)
    lane.from_original = False

    # print lane using sliding window method only
    input = r"./test_images/"
    output =  r'./output_images/5_slidingwindow_lane/'
    #save_images(input, output, process_func=lane.print_lane_display, show_input=False, show=False, reset=lane.reset)
    input = r"./test_videos/"
    output =  r'./output_videos/5_slidingwindow_lane/'
    #save_videos(input, output, process_func=lane.print_lane_display, show_input=False, show=False, reset=lane.reset)

    # Main output
    input = r"./test_images/"
    output =  r'./output_images/6_main_output/'
    save_images(input, output, process_func=lane.main_pipeline, show_input=False, show=False, reset=lane.reset)
    input = r"./test_videos/"
    output =  r'./output_videos/6_main_output/'
    save_videos(input, output, process_func=lane.main_pipeline, show_input=False, show=False, save_frame=[], reset=lane.reset)

    print('The End')
