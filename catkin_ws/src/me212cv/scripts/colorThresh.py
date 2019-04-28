#!/usr/bin/python

# 2.12 Lab 7 object detection: a node for color thresholding
# Jacob Guggenheim 2019
# Jerry Ng 2019

import rospy
import numpy as np
import cv2  # OpenCV module
print cv2.__version__
import imutils
from matplotlib import pyplot as plt
import time
from Tkinter import *
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rospy.init_node('colorThresh', anonymous=True)

# Bridge to convert ROS Image type to OpenCV Image type
cv_bridge = CvBridge()
tk = Tk()
l_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, lower', orient = HORIZONTAL)
l_b.pack()
u_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, upper', orient = HORIZONTAL)
u_b.pack()
u_b.set(255)
l_g = Scale(tk, from_ = 0, to = 255, label = 'Green, lower', orient = HORIZONTAL)
l_g.pack()
u_g = Scale(tk, from_ = 0, to = 255, label = 'Green, upper', orient = HORIZONTAL)
u_g.pack()
u_g.set(255)
l_r = Scale(tk, from_ = 0, to = 255, label = 'Red, lower', orient = HORIZONTAL)
l_r.pack()
u_r = Scale(tk, from_ = 0, to = 255, label = 'Red, upper', orient = HORIZONTAL)
u_r.pack()
u_r.set(255)

l_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, lower', orient = HORIZONTAL)
l_h.pack()
u_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, upper', orient = HORIZONTAL)
u_h.pack()
u_h.set(255)
l_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, lower', orient = HORIZONTAL)
l_s.pack()
u_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, upper', orient = HORIZONTAL)
u_s.pack()
u_s.set(255)
l_v = Scale(tk, from_ = 0, to = 255, label = 'Value, lower', orient = HORIZONTAL)
l_v.pack()
u_v = Scale(tk, from_ = 0, to = 255, label = 'Value, upper', orient = HORIZONTAL)
u_v.pack()
u_v.set(255)


def main():
    rospy.Subscriber('/usb_cam/image_raw', Image, colorThreshCallback)
    print("Subscribing")
    mainloop()
    # rospy.spin()

def colorThreshCallback(msg):
    # convert ROS image to opencv format

    i = 0
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    cv_image = cv_image[0:720, 240:1070]
    #cv_image = cv2.imread('/home/robot/Desktop/image_raw_screenshot_26.04.2019.png')

    # visualize it in a cv window
    cv2.imshow("Original_Image", cv_image)
    cv2.waitKey(3)
    ################ RGB THRESHOLDING ####################
    #get threshold values
    
    lower_bound_RGB = np.array([l_b.get(), l_g.get(), l_r.get()])
    upper_bound_RGB = np.array([u_b.get(), u_g.get(), u_r.get()])

    # threshold
    mask_RGB = cv2.inRange(cv_image, lower_bound_RGB, upper_bound_RGB)

    # get display image
    disp_image_RGB = cv2.bitwise_and(cv_image,cv_image, mask= mask_RGB)
    # cv2.imshow("RGB_Thresholding", disp_image_RGB)
    # cv2.waitKey(3)


    ################ HSV THRESHOLDING ####################
    # conver to HSV
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # get threshold values
    lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
    upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])

    # lower_bound_HSV = np.array([15,60,124])
    # upper_bound_HSV = np.array([30,180,161])

    # threshold
    mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

    # get display image
    disp_image_HSV = cv2.bitwise_and(cv_image,cv_image, mask= mask_HSV)
    cv2.imshow("HSV_Thresholding", disp_image_HSV)
    cv2.waitKey(3)

    ######################CENTROID DETECTION#############################

    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    image = disp_image_HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)

    cv2.imshow("new image", thresh)
    cv2.waitKey(3)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c)>200:
            # compute the center of the contour
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(3)

    #############CIRCLE DETECTION######################
    # gray = cv2.threshold(blurred, l_b.get(), u_b.get(), cv2.THRESH_BINARY)[1]
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(cv_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(cv_image, center, radius, (255, 0, 255), 3) #dictionary of contours with locations and areas of each circle
            #if two circles are close in coordionates (less than small area of olive) than delete smaller one and

    cv2.imshow("detected circles", cv_image)
    cv2.waitKey(3)


if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptionException:
        pass
