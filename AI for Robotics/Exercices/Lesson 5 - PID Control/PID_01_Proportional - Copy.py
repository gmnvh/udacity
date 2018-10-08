# -----------
# User Instructions
#
# Implement a P controller by running 100 iterations
# of robot motion. The desired trajectory for the 
# robot is the x-axis. The steering angle should be set
# by the parameter tau so that:
#
# steering = -tau * crosstrack_error
#
# You'll only need to modify the `run` function at the bottom.
# ------------
 
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 
# this is the Robot class
#

class Robot(object):
    def __init__(self, length=20.0):
        """
        Creates robot and initializes location/orientation to 0, 0, 0.
        """
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.length = length
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.steering_drift = 0.0

    def set(self, x, y, orientation):
        """
        Sets a robot coordinate.
        """
        self.x = x
        self.y = y
        self.orientation = orientation % (2.0 * np.pi)

    def set_noise(self, steering_noise, distance_noise):
        """
        Sets the noise parameters.
        """
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):
        """
        Sets the systematical steering drift parameter
        """
        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):
        """
        steering = front wheel steering angle, limited by max_steering_angle
        distance = total distance driven, most be non-negative
        """
        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift
        steering2 += self.steering_drift

        # Execute motion
        turn = np.tan(steering2) * distance2 / self.length

        if abs(turn) < tolerance:
            # approximate by straight line motion
            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # approximate bicycle model for motion
            radius = distance2 / turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)

############## ADD / MODIFY CODE BELOW ####################
# ------------------------------------------------------------------------
#
# run - does a single control run
robot = Robot()
robot.set(0.0,1.0, 0.0)
robot.set_steering_drift(10.0/180 * np.pi)
#robot.set_noise(0.001, 0.001)
import math
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
def run(robot, tau, dtau, tau_i, n=100, speed=1.0):
    x_trajectory = []
    y_trajectory = []
    # TODO: your code here
    pcte = robot.y
    s_cte = 0
    for t in range(n):
        cte = robot.y
        s_cte += cte
        ster = (-tau * cte) - dtau * (cte -pcte) - tau_i * (s_cte)
        pcte = cte
        #print 'cte', cte
        #print 'ster', ster
        robot.move(ster, speed)
        x_trajectory.append(robot.x)
        y_trajectory.append(robot.y)
        #print 'x',robot.x
        #print 'y', robot.y
        #print 'o', robot.orientation *180/math.pi
        #a = raw_input()
        ax1.plot(robot.x, robot.y, 'b-', label='P controller')
        
        # Create orientation arrow
        ARROW_SIZE = 1.8
        ARROW_WIDTH = 0.6
        ax_xlim = ax1.get_xlim()
        ax_ylim = ax1.get_ylim()
        arrow_xsize = ARROW_SIZE * (ax_xlim[1] - ax_xlim[0]) / 50
        arrow_xsize *= math.cos(robot.orientation)
        arrow_ysize = ARROW_SIZE * (ax_ylim[1] - ax_ylim[0]) / 50
        arrow_ysize *= math.sin(robot.orientation)
        arrow_width = min(arrow_xsize, arrow_ysize) * ARROW_WIDTH / 2

        arrow = plt.Arrow(robot.x, robot.y, arrow_xsize, arrow_ysize, alpha=0.8,
                          facecolor='#356989', edgecolor='#356989', width=arrow_width)
        ax1.add_patch(arrow)
        #plt.show()
        print robot
        
    return x_trajectory, y_trajectory
    
x_trajectory, y_trajectory = run(robot, 0.1, 3.0, 0.004, n=1000)
n = len(x_trajectory)


ax1.plot(x_trajectory, y_trajectory, 'g*', label='P controller')
ax1.plot(x_trajectory, np.zeros(n), 'r', label='reference')

plt.show()