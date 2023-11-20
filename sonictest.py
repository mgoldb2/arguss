from pyfirmata import Arduino, util
import time
import numpy as np
import cv2
from pynput import mouse
"""
trig in 11
echo in 12
"""
global click
click = True
def on_click(x, y, button, pressed):
    global click
    if pressed:
        click = True

global prevx

listener = mouse.Listener(on_click=on_click)
listener.start()
print("Running...")

board = Arduino('/dev/cu.usbmodem14201')
it = util.Iterator(board)
it.start()

print("Arduino ready")

light = board.get_pin('d:13:o')
sonicout = board.get_pin('d:11:o')
sonicin = board.get_pin('d:12:i')

sonicon = False
sonicstart = 0
sonicend = 0
sonictime = 0

prevstate = False

while True:
    if sonicon:
        sonicout.write(0)
        if sonicin.read():
            sonicon = False
            print("got it!")
            sonicend = time.time()
            sonictime = sonicend - sonicstart
            print(sonictime)
    else:
        #print(sonicin.read())
        sonicout.write(1)

    if not prevstate == sonicon:
        prevstate = sonicon
        print("now ", sonicon)

    if click: # clicked
        click = False
        print("You pressed space or clicked!")
        sonicon = True
        sonicstart = time.time()
        sonicout.write(0)
        light.write(0)
    else:
        light.write(1)
