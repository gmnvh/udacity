import os
import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from matplotlib.pyplot import gray

#----------------------------------------------------------------------------------------------------
# Global configuration
#----------------------------------------------------------------------------------------------------

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

def save_images(img_dir, out_dir, process_func=None, show=True, show_input=False, resize=0.7):
    image_list = os.listdir(img_dir)

    for image_file in image_list:
        img = cv2.imread(img_dir + image_file, cv2.IMREAD_COLOR)
        img_input = img.copy()

        # process image if callback function is defined
        output_file = out_dir + image_file
        if process_func is not None:
            img = process_func(img)
            img = img_1C_to_3C(img)

        # write image to file
        cv2.imwrite(output_file, img)
        print('File saved:', output_file)

        # show input image in the left and process in the right
        if show_input is True:
            img = np.concatenate((img_input, img), axis=1)

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

def save_videos(video_dir, out_dir, process_func=None, show=True, show_input=False, resize=0.7):
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
        out = cv2.VideoWriter(output_file, fourcc, fps, size)

        if show is True:
            cv2.namedWindow(output_file)
            cv2.moveWindow(output_file, 0, 0)

        # process video
        ret, frame = cap.read()
        while ret is True:
            start_time = time.time()
            frame_input = frame.copy()

            # process frame
            if process_func is not None:
                frame = process_func(frame)
                frame = img_1C_to_3C(frame)

            # save output video
            out.write(frame)

            # show input image in the left and process in the right
            if show_input is True:
                frame = np.concatenate((frame_input, frame), axis=1)

            # show video
            if show is True:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
                cv2.imshow(output_file, frame)

                elapsed_time = time.time() - start_time
                #cfps = max(0, int(fps - elapsed_time * 100)) - 10
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # read next frame
            ret, frame = cap.read()

        out.release()
        cv2.destroyWindow(output_file)

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

def equalize_test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    return gray

class Lane():
    camera_mtx = None
    camera_dist = None
    poly_src = None
    poly_dst = None
    poly_han = None

    def undistort(self, image):
        return cv2.undistort(image, self.camera_mtx, self.camera_dist, None, self.camera_mtx)


    def thresh_pipeline(self, img):
        '''
        Threshold pipeline to return a binary image (only 1 channel). This function must be used
        by the code pipeline but can not be used for display. The save_videos function needs a 
        3 channel image as a return. For that use `thresh_pipeline_display` function.
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        s_binary = hsl_thresh(img, s_thresh=(100, 255))
        sx_binary = abs_sobel_thresh(gray, 'x', 5, (30, 100))
        mag_binary = mag_thresh(gray, mag_thresh=(100, 255))
        dir_binary = dir_threshold(gray, thresh=(0.7, 1.3))

        combined_binary = np.zeros_like(dir_binary)
        combined_binary[(sx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

        #combined_binary = hsl_grad_thresh(img)
        #combined_binary = self.equalize(img)
        return combined_binary
    
    def thresh_pipeline_display(self, img):
        '''
        Threshold pipeline to return a binary image with 3 channels. This function must be used
        only to generate display images or videos of the threshold process. Must not be used by
        the main lane detection pipeline. For that use `thresh_pipeline` function.
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        s_binary = hsl_thresh(img, s_thresh=(100, 255))
        sx_binary = abs_sobel_thresh(gray, 'x', 5, (30, 100))
        mag_binary = mag_thresh(gray, mag_thresh=(100, 255))
        dir_binary = dir_threshold(gray, thresh=(0.7, 1.3))

        combined = np.zeros_like(dir_binary)
        combined[(sx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((s_binary, combined, combined)) * 255

        return color_binary

    def perspective_display(self, img):
        roi = Image.fromarray(img)
        d = ImageDraw.Draw(roi)
        d.polygon(self.poly_src, fill=None, outline=(20,20,255))
        return np.asarray(roi)

    def warp(self, img):
        if self.poly_han is None:
            self.poly_han = cv2.getPerspectiveTransform(self.poly_src, self.poly_dst)
            print("Perspective calculated")

        thresh = self.thresh_pipeline(img)
        warped = cv2.warpPerspective(thresh, self.poly_han, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def hist(self, img):
        # TO-DO: Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0]//2:,:]
    
        # TO-DO: Sum across image pixels vertically - make sure to set `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)
        
        return histogram
    
    def hist_display(self, img):
        
        img = self.warp(img)
        hist = self.hist(img)
        print(img.shape)
        fig = plt.figure(figsize=(12.8,7.2))
        xs = np.arange(img.shape[0])
        plt.plot(hist)
        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        print(img.shape)
        return img
#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #######
    # DELETE THIS BEFORE SUBMIT
    import os
    os.chdir(r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 2 - Advanced Lane Finding')
    #os.chdir(r'C:\Gustavo\projects\udacity\Self-Driving Cars\github\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 2 - Advanced Lane Finding')
    print(os.path.abspath("./"))
    #######
    #mtx, dist = camera_calibration()
    # show_camera_calibration(mtx, dist, True)

    # create lane object and initialize camera parameters
    lane = Lane()
    #lane.camera_mtx = mtx
    #lane.camera_dist = dist

    # undistort images
    input = r"./test_images/"
    output =  r'./output_images/1_undistort/'
    #save_images(input, output, process_func=lane.undistort, show_input=True)
    input = r"./test_videos/"
    output =  r'./output_videos/1_undistort/'
    #save_videos(input, output, process_func=lane.undistort, show_input=False, show=False)
    
    # threshold - colored binary for display
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/2_threshold/'
    #save_images(input, output, process_func=lane.thresh_pipeline_display, show_input=True)
    input = r'./output_videos/1_undistort/'
    output =  r'./output_videos/2_threshold/'
    #save_videos(input, output, process_func=lane.thresh_pipeline_display, show_input=True, show=True)
    
    # threshold - colored binary for display
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/2_threshold/'
    #save_images(input, output, process_func=lane.thresh_pipeline_display, show_input=True)
    input = r'./output_videos/1_undistort/'
    output =  r'./output_videos/2_threshold/'
    #save_videos(input, output, process_func=lane.thresh_pipeline, show_input=True, show=True)
    
    #lane.poly_src = np.float32([[280,660], [1025, 660], [672, 440], [608, 440]])
    #lane.poly_dst = np.float32([[280,660], [1025, 660], [1025, 440], [280, 440]])
    #lane.poly_src = np.float32([[190,720], [1130, 720], [705, 455], [585, 455]])
    #lane.poly_dst = np.float32([[200,720], [1080, 720], [1080, 0], [200, 0]])
    lane.poly_src = np.float32([[190,720], [1130, 720], [705, 455], [585, 455]])
    
    offset = 300
    img_size = (1280, 720)
    lane.poly_dst = np.float32([[offset,img_size[1]], [img_size[0]-offset, 720],
                                [img_size[0]-offset, 0], [offset, 0]])
    
    # perspective display
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/3_perspective/'
    #save_images(input, output, process_func=lane.perspective_display, show_input=True)
    
    # eagle's eye
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/4_eagle_eye/'
    #save_images(input, output, process_func=lane.warp, show_input=True)
    input = r'./output_videos/1_undistort/'
    output =  r'./output_videos/4_eagle_eye/'
    #save_videos(input, output, process_func=lane.warp, show_input=True, show=False)
    
    # histogram
    input = r'./output_images/1_undistort/'
    output =  r'./output_images/5_hist/'
    save_images(input, output, process_func=lane.hist_display, show_input=True)
    input = r'./output_videos/4_eagle_eye/'
    output =  r'./output_videos/5_hist/'
    #save_videos(input, output, process_func=lane.warp, show_input=True, show=False)
    print('The End')
