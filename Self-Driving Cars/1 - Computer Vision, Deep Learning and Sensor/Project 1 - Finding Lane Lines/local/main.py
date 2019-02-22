import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

print("Lane Detection")

# Video file
video_file = r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 1 - Finding Lane Lines\test_videos\solidWhiteRight.mp4'
video_file = r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 1 - Finding Lane Lines\test_videos\20190206_home_to_cnx_vga.mp4'
video_file = r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 1 - Finding Lane Lines\test_videos\20190207_cnx_to_home_vga.mp4'
video_file = r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 1 - Finding Lane Lines\test_videos\challenge.mp4'

video_output = r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 1 - Finding Lane Lines\test_videos_output\local_out.mp4'

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines_original(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_line_slope = []
    left_line_b = []
    right_line_slope = []
    right_line_b = []
    
    if lines is None:
        return
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if abs(1-abs(slope)) > 0.5:
                continue
            b = y1 - (slope*x1)
            if slope < 0:
                left_line_slope.append(slope)
                left_line_b.append(b)
            else:
                right_line_slope.append(slope)
                right_line_b.append(b)
    
    
   
    
    
    
    
    
    if len(left_line_slope) != 0:
        left_avr_slope = sum(left_line_slope)/len(left_line_slope)
        left_avr_b = sum(left_line_b)/len(left_line_b)
    
    # (0,imshape[0]),(410, 330), (580, 330), (imshape[1],imshape[0])]]
        low_roi_y1 = img.shape[0]
        low_roi_x1 = int((low_roi_y1 - left_avr_b)/left_avr_slope)
        
        #high_roi_y2 = 330
        high_roi_y2 = 450
        high_roi_x2 = int((high_roi_y2 - left_avr_b)/left_avr_slope)
        
        cv2.line(img, (low_roi_x1, low_roi_y1), (high_roi_x2, high_roi_y2), [0, 255, 0], 4)
    
    if len(right_line_slope) != 0:
        
        right_avr_slope = sum(right_line_slope)/len(right_line_slope)
        right_avr_b = sum(right_line_b)/len(right_line_b)
    
        low_roi_y1 = img.shape[0]
        low_roi_x1 = int((low_roi_y1 - right_avr_b)/right_avr_slope)
        
        high_roi_y2 = 450
        high_roi_x2 = int((high_roi_y2 - right_avr_b)/right_avr_slope)
        
        cv2.line(img, (low_roi_x1, low_roi_y1), (high_roi_x2, high_roi_y2), [255, 0, 0], 4)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., y=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + y
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, y)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # convert to gray scale
    gray = grayscale(image)
    
    # blur - Kernel size = 3
    blur = gaussian_blur(gray, 3)
    
    # edge detection
    edges = canny(blur, 70, 210)
    
    # region of interest
    imshape = edges.shape
    
    # roi for lesson videos
    #vertices = np.array([[(0,imshape[0]),(410, 330), (580, 330), (imshape[1],imshape[0])]], dtype=np.int32)
    
    # roi for home to cnx video
    #vertices = np.array([[(70,410),(320, 320), (360, 320), (620, 410)]], dtype=np.int32)
    
    # roi for chalenge video
    vertices = np.array([[(20, 680), (610, 450), (720, 450), (1260, 680)]], dtype=np.int32)
    
    roi = region_of_interest(edges, vertices)
    
    # find lines
    rho = 2
    theta = np.pi/180
    threshold = 20 #20
    min_line_length = 60
    max_line_gap = 30
    line_image = np.copy(roi)*0 #creating a blank to draw lines on
    
    lines = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)
 
    x = [point[0] for point in vertices[0]]
    y = [point[1] for point in vertices[0]]
    
    cv2.polylines(lines, vertices, True, 255, 1, 8)
    
    # combine original image with lines detected
    result = cv2.addWeighted(image, 0.8, lines, 1, 0)
    
    
    return result, gray, edges

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
#clip = VideoFileClip(video_file).subclip(0, 20)
#out_clip = clip.fl_image(process_image)
#out_clip.write_videofile(video_output, audio=False)

cap = cv2.VideoCapture(video_file)
print('Video is ', cap.get(3), 'x', cap.get(4))

while(cap.isOpened()):
    ret, frame = cap.read()
    
    cv2.imshow('original', frame)
    frame, gray, edges = process_image(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('edges', edges)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
