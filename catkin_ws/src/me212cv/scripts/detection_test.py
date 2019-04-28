


import numpy as np
import cv2  # OpenCV module
print cv2.__version__
import imutils

cv_image = cv2.imread('/home/robot/Desktop/image_raw_screenshot_26.04.2019.png')

hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

# get threshold values
# lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
# upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])

lower_bound_HSV = np.array([139, 116, 0])
upper_bound_HSV = np.array([225, 225, 225])

# threshold
mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

# get display image
disp_image_HSV = cv2.bitwise_and(cv_image, cv_image, mask=mask_HSV)
#cv2.imshow("HSV_Thresholding", disp_image_HSV)
#cv2.waitKey(0)

# load the image, convert it to grayscale, blur it slightly,
    # and threshold it
image = disp_image_HSV

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)[1]
#thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)

    #cv2.imshow("new image", thresh)
    #cv2.waitKey(3)

    # find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

    # loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)