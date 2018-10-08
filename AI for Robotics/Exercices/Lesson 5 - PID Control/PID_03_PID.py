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

def run(robot, tau_p, tau_d, tau_i, n=1000, speed=1.0):
    x_trajectory = []
    y_trajectory = []
    ster_ang = []
    # TODO: your code here
    pcte = robot.y
    s_cte = 0
    for t in range(n):
        cte = robot.y
        s_cte += cte
        ster = -tau_p * cte - tau_d * (cte - pcte) - tau_i * (s_cte)
        pcte = cte
        robot.move(ster, speed)
        x_trajectory.append(robot.x)
        y_trajectory.append(robot.y)
        ster_ang.append(ster)
    return x_trajectory, y_trajectory, ster_ang


# Prepare graph
fig, axes = plt.subplots(2, 3, figsize=(8, 8))

# Run with tau_p 0.1 and tau_d 3.0
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.1, 3.0, 0)
n = len(x_trajectory)

# Print values for tau_p 0.1 and tau_d 3.0
for x, y, ster in zip(x_trajectory, y_trajectory, ster_ang):
    print str(x) + ', ' + str(y) + ', ' + str(ster)

# Add to graph
axes[0, 0].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[0, 0].plot(x_trajectory, np.zeros(n), 'r', label='reference')

# Run with tau_p 0.3 tau_d 3.0
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.3, 3.0, 0)
n = len(x_trajectory)

# Add to graph
axes[1, 0].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[1, 0].plot(x_trajectory, np.zeros(n), 'r', label='reference')

# Run with tau_p 0.1 tau_d 6.0
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.1, 6.0, 0)
n = len(x_trajectory)

# Add to graph
axes[0, 1].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[0, 1].plot(x_trajectory, np.zeros(n), 'r', label='reference')

# Run with tau_p 0.3 tau_d 6.0
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.3, 6.0, 0)
n = len(x_trajectory)

# Add to graph
axes[1, 1].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[1, 1].plot(x_trajectory, np.zeros(n), 'r', label='reference')






# Run with tau_p 0.1 tau_d 6.0 tau_i 0.01
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.3, 1, 0.0003)
n = len(x_trajectory)

# Add to graph
axes[0, 2].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[0, 2].plot(x_trajectory, np.zeros(n), 'r', label='reference')

# Run with tau_p 0.3 tau_d 6.0 tau_i 0.1
robot.set(0.0, 1.0, 0.0)
x_trajectory, y_trajectory, ster_ang = run(robot, 0.3, 0.0, 0)
n = len(x_trajectory)

# Add to graph
axes[1, 2].plot(x_trajectory, y_trajectory, 'g', label='P controller')
axes[1, 2].plot(x_trajectory, np.zeros(n), 'r', label='reference')