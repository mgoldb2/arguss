from pyfirmata import Arduino, util
import time
import numpy as np
import cv2

#green yellow blue with button facing left

print("Running...")
#this value is 14101 for left USB port and 14201 for right, in my experience
board = Arduino('/dev/cu.usbmodem14101')
it = util.Iterator(board)
it.start()

print("Arduino ready")

mosfet = board.get_pin('d:3:o')
#0 is on for raw light, 0 dims through MOSFET
while True:
    print(0)
    mosfet.write(0)
    time.sleep(5)
    print(1)
    mosfet.write(1)
    time.sleep(3)
