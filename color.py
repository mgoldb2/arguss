from pyfirmata import Arduino, util
import time
import numpy as np
import cv2

board = Arduino('/dev/cu.usbmodem14201')
it = util.Iterator(board)
it.start()

knob = board.get_pin('a:0:i')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()
     # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #lower_blue = np.array([10,50,50])
    #upper_blue = np.array([30,255,255])

    lower_color = np.array([knob.read() * 160,0,0])
    upper_color = np.array([knob.read() * 180 + 20,255,255])


    #blue
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])

    #color finder
    #mycolor = np.uint8([[[0,180,255]]])
    #hsv_mycolor = cv2.cvtColor(mycolor,cv2.COLOR_BGR2HSV)
    #print(hsv_mycolor)

    # Threshold the HSV image to get only intended colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    key = cv2.waitKey(20)
    if key == 32: # print values
        print("[", knob.read() * 160, ",50,50]")
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
"""
light = board.get_pin('d:11:p')
button = board.get_pin('a:0:i')
while True:
    if (button.read() == 1.0):
        light.write(0)
    else:
        light.write(1)
    board.pass_time(0.1)
    #time.sleep(0.3)
    #board.pass_time(0.5)
    #light.write(0)
    #time.sleep(0.3)
"""
#board.analog[0].enable_reporting()
#board.analog[0].read()
#board.digital[13].write(1)
