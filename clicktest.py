from pyfirmata import Arduino, util
import time
import numpy as np
import cv2
from pynput import mouse

global click
click = True
global dragx
dragx = 0
def on_click(x, y, button, pressed):
    global click
    if pressed:
        click = True

global prevx
global mousex
global mousey
def on_move(x, y):
    global mousex
    global mousey
    mousex = x
    mousey = y
    global prevx
    global dragx
    try:
        dragx = prevx - x
    except:
        prevx = x
    prevx = x

listener = mouse.Listener(on_click=on_click,on_move=on_move)
listener.start()
print("Running...")
#print(mouse.position())
board = Arduino('/dev/cu.usbmodem14201')
it = util.Iterator(board)
it.start()

print("Arduino ready")

light = board.get_pin('d:13:o')
servo = board.get_pin('d:9:s')

servopos = 0

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()

    cv2.imshow("frame",frame)

    servo.write(servopos)

    servopos += dragx
    if servopos > 180:
        servopos = 180
    if servopos < 0:
        servopos = 0
    if dragx != 0:
        print(servopos)
        #dragx = 0

    key = cv2.waitKey(20)
    dirkey = cv2.waitKey(5)
    if key == 32 or click: #pressed space or clicked
        click = False
        print("You pressed space or clicked!")
        light.write(0)
    else:
        light.write(1)
    if dirkey == 97: #pressed a
        servopos += 10
        if servopos > 180:
            servopos = 180
        print(servopos)
    if dirkey == 100: #pressed d
        servopos -= 10
        if servopos < 0:
            servopos = 0
        print(servopos)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
