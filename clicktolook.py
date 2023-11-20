from pyfirmata import Arduino, util
import time
import numpy as np
import cv2
from pynput import mouse

global click
click = True
def on_click(x, y, button, pressed):
    global click
    if pressed:
        click = True
global mousex
mousex = 0
global mousey
mousey = 0
def on_move(x, y):
    global mousex
    global mousey
    mousex = x
    mousey = y

listener = mouse.Listener(on_click=on_click,on_move=on_move)
listener.start()
print("Running...")
#this value is 14101 for left USB port and 14201 for right
board = Arduino('/dev/cu.usbmodem14101')
it = util.Iterator(board)
it.start()

print("Arduino ready")

light = board.get_pin('d:13:o')
servoh = board.get_pin('d:9:s')
servov = board.get_pin('d:10:s')

servohpos = 0
servovpos = 0

refPt = []

def cv2click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    global servohpos
    global servovpos

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print("This is a cv2 click!")
        refPt = (x, y)
        servohpos += (frame.shape[1] / 2 - refPt[0]) * 0.1
        if servohpos < 0:
            servohpos = 0
        elif servohpos > 180:
            servohpos = 180
        print(servohpos)
        servoh.write(servohpos)
        servovpos -= (frame.shape[0] / 2 - refPt[1]) * 0.1
        if servovpos < 0:
            servovpos = 0
        elif servovpos > 180:
            servovpos = 180
        print(servovpos)
        servov.write(servovpos)

cv2.namedWindow("frame")
vc = cv2.VideoCapture(0)
cv2.setMouseCallback("frame", cv2click)

reticle = cv2.imread("reticle.png")

def blend_non_transparent(face_img, overlay_img):
    #resize overlay to fit background
    overlay_img = cv2.resize(overlay_img,(face_img.shape[1],face_img.shape[0]))
    # Let's find a mask covering all the non-black (foreground) pixels
    # NB: We need to do this on grayscale version of the image
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

'''
# We load the images
face_img = cv2.imread("lena.png", -1)
overlay_img = cv2.imread("overlay.png", -1)

result_1 = blend_non_transparent(face_img, overlay_img)
cv2.imwrite("merged.png", result_1)
'''

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    cv2.moveWindow("frame",0,0)
else:
    rval = False

while rval and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) >= 1:
    #global mousex
    #global mousey
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()

    #this adds the reticle overlay
    frame = blend_non_transparent(frame, reticle)

    #draws a rectangle at the mouse position
    #cv2.rectangle(frame, (0, 0), (int(mousex), int(mousey) - 75), (255,0,0), 2)

    #moves the window to the top left corner of the screen
    #cv2.moveWindow("frame",0,0)

    cv2.imshow("frame",frame)

    #servo.write(servopos)

    key = cv2.waitKey(50)
    '''
    dirkey = cv2.waitKey(100)
    if key == 32 or click: #pressed space or clicked
        click = False
        print("You pressed space or clicked!")
        light.write(0)
        #servovpos -= ((frame.shape[0] / 2) / mousey) * 180
        #if servovpos < 0:
        #    servovpos = 0
        #elif servovpos > 180:
        #    servovpos = 180
        #print(servovpos)
    else:
        light.write(1)
    if dirkey == 97: #pressed a
        servohpos += 5
        if servohpos > 180:
            servohpos = 180
        print(servohpos)
        servoh.write(servohpos)
    if dirkey == 100: #pressed d
        servohpos -= 5
        if servohpos < 0:
            servohpos = 0
        print(servohpos)
        servoh.write(servohpos)
    if dirkey == 115: #pressed s
        servovpos += 5
        if servovpos > 180:
            servovpos = 180
        print(servovpos)
        servov.write(servovpos)
    if dirkey == 119: #pressed w
        servovpos -= 5
        if servovpos < 0:
            servovpos = 0
        print(servovpos)
        servov.write(servovpos)
    '''
    if key == 27: # exit on ESC
        servoh.write(0)
        servov.write(0)
        break
cv2.destroyAllWindows()
