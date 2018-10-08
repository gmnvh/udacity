# -----------------
# USER INSTRUCTIONS
#
# Write a function in the class robot called move()
#
# that takes self and a motion vector (this
# motion vector contains a steering* angle and a
# distance) as input and returns an instance of the class
# robot with the appropriate x, y, and orientation
# for the given motion.
#
# *steering is defined in the video
# which accompanies this problem.
#
# For now, please do NOT add noise to your move function.
#
# Please do not modify anything except where indicated
# below.
#
# There are test cases which you are free to use at the
# bottom. If you uncomment them for testing, make sure you
# re-comment them before you submit.

import random
from math import *

# --------
# 
# the "world" has 4 landmarks.
# the robot's initial coordinates are somewhere in the square
# represented by the landmarks.
#
# NOTE: Landmark coordinates are given in (y, x) form and NOT
# in the traditional (x, y) format!

landmarks  = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] # position of 4 landmarks
world_size = 100.0 # world is NOT cyclic. Robot is allowed to travel "out of bounds"
max_steering_angle = pi/4 # You don't need to use this value, but it is good to keep in mind the limitations of a real car.

# ------------------------------------------------
# 
# this is the robot class
#

class robot:

    # --------

    # init: 
    #	creates robot and initializes location/orientation 
    #

    def __init__(self, length = 10.0):
        self.x = random.random() * world_size # initial x position
        self.y = random.random() * world_size # initial y position
        self.orientation = random.random() * 2.0 * pi # initial orientation
        self.length = length # length of robot
        self.bearing_noise  = 0.0 # initialize bearing noise to zero
        self.steering_noise = 0.0 # initialize steering noise to zero
        self.distance_noise = 0.0 # initialize distance noise to zero
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))
    # --------
    # set: 
    #	sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)


    # --------
    # set_noise: 
    #	sets the noise parameters
    #

    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.bearing_noise  = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)
    
    ############# ONLY ADD/MODIFY CODE BELOW HERE ###################

    # --------
    # move:
    #   move along a section of a circular path according to motion
    #
    
    def move(self, motion, tolerance=0.001): # Do not change the name of this function

        str_ang = motion[0]
        distance = motion[1]
        
        distance = random.gauss(distance, self.distance_noise)
        str_ang = random.gauss(str_ang, self.steering_noise)

        turning_ang = tan(str_ang) * distance / self.length

        if (abs(turning_ang) > tolerance):
            curve_r = distance / turning_ang
            cx = self.x - (sin(self.orientation) * curve_r)
            cy = self.y + (cos(self.orientation) * curve_r)
            orientation = (self.orientation + turning_ang) % (2.0*pi)

            x = cx + (sin(orientation) * curve_r)
            y = cy - (cos(orientation) * curve_r)

        else:
            x = self.x + (distance * cos(self.orientation))
            y = self.y + (distance * sin(self.orientation))
            orientation = (self.orientation + turning_ang) % (2.0*pi)

        result = robot(self.length)
        result.set(x, y, orientation)
        result.set_noise(bearing_noise, steering_noise, distance_noise)

        return result # make sure your move function returns an instance
                      # of the robot class with the correct coordinates.

    def sense(self, add_noise = 1): #do not change the name of this function
        Z = []

        for landmark in landmarks:
            l_x = landmark[1]
            l_y = landmark[0]

            d_x = l_x - self.x
            d_y = l_y - self.y
            b = atan2(d_y, d_x) - self.orientation
            b %= 2.0 * pi

            if add_noise:
                b += random.gauss(0.0, self.bearing_noise)

            Z.append(b)

        return Z
    ############## ONLY ADD/MODIFY CODE ABOVE HERE ####################
        

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program provides its own code for testing your
## move function with randomized motion data.

## --------
## TEST CASE:
## 
## 1) The following code should print:
##       Robot:     [x=0.0 y=0.0 orient=0.0]
##       Robot:     [x=10.0 y=0.0 orient=0.0]
##       Robot:     [x=19.861 y=1.4333 orient=0.2886]
##       Robot:     [x=39.034 y=7.1270 orient=0.2886]
##
##
length = 20.
bearing_noise  = 0.0
steering_noise = 0.0
distance_noise = 0.0
#
myrobot = robot(length)
myrobot.set(0.0, 0.0, 0.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)
#
motions = [[0.0, 10.0], [pi / 6.0, 10.], [0.0, 20.0]]
#
T = len(motions)

print 'Robot:    ', myrobot
for t in range(T):
    myrobot = myrobot.move(motions[t])
    print 'Robot:    ', myrobot

#

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program provides its own code for testing your
## move function with randomized motion data.

    
## 2) The following code should print:
##      Robot:     [x=0.0 y=0.0 orient=0.0]
##      Robot:     [x=9.9828 y=0.5063 orient=0.1013]
##      Robot:     [x=19.863 y=2.0201 orient=0.2027]
##      Robot:     [x=29.539 y=4.5259 orient=0.3040]
##      Robot:     [x=38.913 y=7.9979 orient=0.4054]
##      Robot:     [x=47.887 y=12.400 orient=0.5067]
##      Robot:     [x=56.369 y=17.688 orient=0.6081]
##      Robot:     [x=64.273 y=23.807 orient=0.7094]
##      Robot:     [x=71.517 y=30.695 orient=0.8108]
##      Robot:     [x=78.027 y=38.280 orient=0.9121]
##      Robot:     [x=83.736 y=46.485 orient=1.0135]
##
##

length = 20.
bearing_noise  = 0.0
steering_noise = 0.0
distance_noise = 0.0
##
myrobot = robot(length)
myrobot.set(0.0, 0.0, 0.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)
##
motions = [[0.5, 10.] for row in range(10)]
##
T = len(motions)
##
points = []
print 'Robot:    ', myrobot
for t in range(T):
    points.append([myrobot.x, myrobot.y, myrobot.orientation])
    myrobot = myrobot.move(motions[t])
    print 'Robot:    ', myrobot

from VehicleMovePlot import VehicleMovePlot
import matplotlib.pyplot as plt

myplot = VehicleMovePlot(points, interval=1000)
plt.show()

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program providels its own code for testing your
## move function with randomized motion data.
myrobot = robot(length)
myrobot.set(30.0, 20.0, pi / 5.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)
print myrobot.sense()

