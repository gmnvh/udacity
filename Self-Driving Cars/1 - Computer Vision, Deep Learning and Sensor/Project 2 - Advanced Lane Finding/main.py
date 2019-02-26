import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#############################
# Global configuration
#############################

# Project camera
calibration_images = r'./camera_cal'
calibration_test_image = r'./camera_cal/calibration1.jpg'
chessboard_corner_x = 9
chessboard_corner_y = 6
out_camera_calibration = r'./examples/camera_calibration.jpg'

# My camera

#############################

#############################
# Camera calibration
#############################

def camera_calibration():
    '''
    '''

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

if __name__ == '__main__':
    #######
    # DELETE THIS BEFORE SUBMIT
    import os
    os.chdir(r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 2 - Advanced Lane Finding')
    print(os.path.abspath("./"))
    #######
    mtx, dist = camera_calibration()
    show_camera_calibration(mtx, dist, True)