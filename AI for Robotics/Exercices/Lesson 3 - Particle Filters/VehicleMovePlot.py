"""
Animation of moviment of the vehicle.

Author: Gustavo Muller Nunes
"""
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class VehicleMovePlot(object):
    '''
        Show animated move of a list of points or update plot with real time
        data.

        Arguments:
            points: list of points with 3 columns [x, y, orientation]
    '''
    def __init__(self, points=None, interval=100):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, xlim=(-10, 100), ylim=(-10, 100))
        self.ax.grid(True)

        if points is not None:
            self.points = np.array(points)
            self.ani = animation.FuncAnimation(self.fig, self.animate,
                                               frames=len(points),
                                               interval=interval, blit=True)

    def animate(self, i):
        x = self.points[0:i+1, 0]
        y = self.points[0:i+1, 1]
        orientation = self.points[0:i+1, 2]

        # Create previous points
        self.p1, = self.ax.plot(x, y, 'bo', ms=5, alpha=0.3, color='#4588b2')

        # Create actual point
        self.p2, = self.ax.plot(x[-1], y[-1], 'bo', ms=10, alpha=0.7, color='#4588b2')

        # Create orientation arrow
        ARROW_SIZE = 1.8
        ARROW_WIDTH = 0.6
        ax_xlim = self.ax.get_xlim()
        ax_ylim = self.ax.get_ylim()
        arrow_xsize = ARROW_SIZE * (ax_xlim[1] - ax_xlim[0]) / 50
        arrow_xsize *= math.cos(orientation[-1])
        arrow_ysize = ARROW_SIZE * (ax_ylim[1] - ax_ylim[0]) / 50
        arrow_ysize *= math.sin(orientation[-1])
        arrow_width = min(arrow_xsize, arrow_ysize) * ARROW_WIDTH / 2

        arrow = plt.Arrow(x[-1], y[-1], arrow_xsize, arrow_ysize, alpha=0.8,
                          facecolor='#356989', edgecolor='#356989', width=arrow_width)
        self.p3 = self.ax.add_patch(arrow)

        return self.p1, self.p2, self.p3

    def show(self):
        plt.show()

    def reset(self):
        pass

    def update(self, points):
        self.points = np.array(points)
        x = self.points[0:, 0]
        y = self.points[0:, 1]
        orientation = self.points[0:, 2]

        # Create previous points
        p1, = self.ax.plot(x, y, 'bo', ms=5, alpha=0.3, color='#4588b2')

        # Create actual point
        p2, = self.ax.plot(x[-1], y[-1], 'bo', ms=10, alpha=0.7, color='#4588b2')

        # Create orientation arrow
        ax_xlim = self.ax.get_xlim()
        ax_ylim = self.ax.get_ylim()
        arrow_xsize = 2* (ax_xlim[1] - ax_xlim[0]) / 50
        arrow_xsize *= math.cos(math.pi/2 + orientation[-1])
        arrow_ysize = 2* (ax_ylim[1] - ax_ylim[0]) / 50
        arrow_ysize *= math.sin(math.pi/2 + orientation[-1])

        arrow = plt.Arrow(x[-1], y[-1], arrow_xsize, arrow_ysize, alpha=0.8,
                          facecolor='#356989', edgecolor='#356989')
        p3 = self.ax.add_patch(arrow)

        plt.draw()
        return p1, p2, p3
