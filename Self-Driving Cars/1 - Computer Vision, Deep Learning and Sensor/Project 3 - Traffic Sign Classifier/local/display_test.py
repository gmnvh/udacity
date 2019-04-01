import os
import logging

os.chdir(r'G:\cnx\projects\udacity\Self-Driving Cars\1 - Computer Vision, Deep Learning and Sensor\Project 3 - Traffic Sign Classifier\local')

import Display

FORMAT = '%(module)-15s:%(levelname)-5s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger("display").setLevel(logging.DEBUG)

logging.info("Main file")
logging.debug("Test")

input = r'G:/cnx/projects/udacity/Self-Driving Cars/1 - Computer Vision, Deep Learning and Sensor/Project 2 - Advanced Lane Finding/test_images'
output = r'G:/test/'
a = Display.Image(input, output)
a.run()
