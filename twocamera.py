from pyfirmata import Arduino, util
import time
import numpy as np
import cv2
import math

print("Running...")
#this value is 14101 for left USB port and 14201 for right, in my experience
board = Arduino('/dev/cu.usbmodem14101')
it = util.Iterator(board)
it.start()

print("Arduino ready")

#light (unnecessary)
#light = board.get_pin('d:13:o')
servoh = board.get_pin('d:9:s')
servov = board.get_pin('d:10:s')

servohpos = 70
servovpos = 160
servoh.write(servohpos)
servov.write(servovpos)

dragging = False
dragging2 = False

beginRefPt = []
endRefPt = []

mousex = 0
mousey = 0

track_window = (0, 0, 0, 0)
track_window2 = (0, 0, 0, 0)
roi_hist = 0
#roi_hist2 = 0
is_tracking = False
is_tracking2 = False

targethue = 0
huemin = 0
huemax = 0
targetsat = 0
satmin = 0
satmax = 0
targetval = 0
valmin = 0
valmax = 0

def cv2click(event, x, y, flags, param):
    # grab references to the global variables
    global beginRefPt
    global endRefPt
    global servohpos
    global servovpos
    global is_tracking
    global is_tracking2
    global roi_hist
    global track_window
    global track_window2
    global dragging
    global mousex
    global mousey
    global targethue
    global huemin
    global huemax
    global targetsat
    global satmin
    global satmax
    global targetval
    global valmin
    global valmax

    mousex = x
    mousey = y

    #if dragging:
    #    cv2.rectangle(frame, (beginRefPt[0], beginRefPt[1]), (x, y), (0,255,0), 2)

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(frame.mean(axis=0).mean(axis=0))
        dragging = True
        is_tracking = False
        #print("This is a cv2 click!")
        beginRefPt = (x, y)
        try:
            cv2.destroyWindow("selection")
        except:
            None
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        is_tracking = True
        #print("This is a cv2 unclick!")
        endRefPt = (x, y)
        #r, h, c, w = 250,90,400,125
        if beginRefPt[1] > endRefPt[1]:
            r = endRefPt[1]
            h = beginRefPt[1] - endRefPt[1]
        else:
            r = beginRefPt[1]
            h = endRefPt[1] - beginRefPt[1]
        if beginRefPt[0] > endRefPt[0]:
            c = endRefPt[0]
            w = beginRefPt[0] - endRefPt[0]
        else:
            c = beginRefPt[0]
            w = endRefPt[0] - beginRefPt[0]
        if h == 0:
            h = 1
            is_tracking = False
        if w == 0:
            w = 1
            is_tracking = False
        #r, h, c, w = beginRefPt[1], endRefPt[1] - beginRefPt[1], beginRefPt[0], endRefPt[0] - beginRefPt[0]
        track_window = (c, r, w, h)

        if not is_tracking2:
            track_window2 = (c, r, w, h)
            is_tracking2 = True


        # set up the ROI for tracking
        dummy, rawframe = vc.read()
        roi = rawframe[r:r+h, c:c+w]
        #print(roi.mean(axis=0).mean(axis=0))
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        #this is for average color
        #targethue = hsv_roi.mean(axis=0).mean(axis=0)[0]

        #this is for dominant color
        targethue, targetsat, targetval = bincount_app(hsv_roi)

        #this is for histogram-based dominant value
        '''
        targetprecision = 10
        targethist = cv2.calcHist([hsv_roi],[0],None,[int(180/targetprecision)],[0,180])
        #cv2.normalize(target_hist,target_hist,0,255,cv2.NORM_MINMAX)
        targethue = int(np.argmax(targethist)*targetprecision)
        targethist = cv2.calcHist([hsv_roi],[1],None,[int(180/targetprecision)],[0,180])
        #cv2.normalize(target_hist,target_hist,0,255,cv2.NORM_MINMAX)
        targetsat = int(np.argmax(targethist)*targetprecision)
        '''

        huemin = targethue - 15
        if huemin < 0:
            huemin = 0
        huemax = targethue + 15
        if huemax > 180:
            huemax = 180
        print("Hue selected is ", targethue)

        satmin = targetsat - 40
        if satmin < 0:
            satmin = 0
        satmax = targetsat + 40
        if satmax > 255:
            satmax = 255
        print("Sat selected is ", targetsat)

        valmin = targetval - 60
        if valmin < 0:
            valmin = 0
        valmax = targetval + 60
        if valmax > 255:
            valmax = 255
        print("Val selected is ", targetval)

        '''
        satmin = 0
        satmax = 255
        targetsat = 128
        valmin = 0
        valmax = 255
        targetval = 128
        '''
        #values set to 60.,32. for fading out darkness
        #mask = cv2.inRange(hsv_roi, np.array((huemin, 30.,22.)), np.array((huemax,255.,230.)))
        #mask = cv2.inRange(hsv_roi, np.array((huemin, 0,0)), np.array((huemax,255,255)))
        mask = cv2.inRange(hsv_roi, np.asarray((huemin,satmin,valmin)), np.asarray((huemax,satmax,valmax)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[30],[0,180])
        #kernel = np.ones((5,1), np.uint8)
        #roi_hist = cv2.dilate(roi_hist, kernel, iterations=1)
        #roi_hist = cv2.blur(roi_hist,(20,1))
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Use this to show the area you selected. Buggy.
        #cv2.namedWindow("selection")
        #cv2.imshow("selection", roi)

        #is_tracking = True

def cv2click2(event, x, y, flags, param):
    # grab references to the global variables
    global beginRefPt
    global endRefPt
    global servohpos
    global servovpos
    global is_tracking
    global is_tracking2
    #global roi_hist2
    global track_window2
    global dragging2
    global mousex
    global mousey
    global targethue
    global huemin
    global huemax
    global targetsat
    global satmin
    global satmax
    global targetval
    global valmin
    global valmax

    mousex = x
    mousey = y

    if is_tracking:
        #if dragging:
        #    cv2.rectangle(frame, (beginRefPt[0], beginRefPt[1]), (x, y), (0,255,0), 2)

        # if the left mouse button was clicked, record the (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            #print(frame.mean(axis=0).mean(axis=0))
            dragging2 = True
            is_tracking2 = False
            #print("This is a cv2 click!")
            beginRefPt = (x, y)
            try:
                cv2.destroyWindow("selection")
            except:
                None
        elif event == cv2.EVENT_LBUTTONUP:
            dragging2 = False
            is_tracking2 = True
            #print("This is a cv2 unclick!")
            endRefPt = (x, y)
            #r, h, c, w = 250,90,400,125
            if beginRefPt[1] > endRefPt[1]:
                r = endRefPt[1]
                h = beginRefPt[1] - endRefPt[1]
            else:
                r = beginRefPt[1]
                h = endRefPt[1] - beginRefPt[1]
            if beginRefPt[0] > endRefPt[0]:
                c = endRefPt[0]
                w = beginRefPt[0] - endRefPt[0]
            else:
                c = beginRefPt[0]
                w = endRefPt[0] - beginRefPt[0]
            if h == 0:
                h = 1
                is_tracking2 = False
            if w == 0:
                w = 1
                is_tracking2 = False
            #r, h, c, w = beginRefPt[1], endRefPt[1] - beginRefPt[1], beginRefPt[0], endRefPt[0] - beginRefPt[0]
            track_window2 = (c, r, w, h)

            # set up the ROI for tracking
            dummy, rawframe = vc2.read()
            roi = rawframe[r:r+h, c:c+w]
            #print(roi.mean(axis=0).mean(axis=0))
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            #values set to 60.,32. for fading out darkness
            #mask = cv2.inRange(hsv_roi, np.array((huemin, 30.,22.)), np.array((huemax,255.,230.)))
            #mask = cv2.inRange(hsv_roi, np.array((huemin, 0,0)), np.array((huemax,255,255)))
            #mask = cv2.inRange(hsv_roi, np.asarray((huemin,satmin,valmin)), np.asarray((huemax,satmax,valmax)))
            #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[30],[0,180])
            #kernel = np.ones((5,1), np.uint8)
            #roi_hist = cv2.dilate(roi_hist, kernel, iterations=1)
            #roi_hist = cv2.blur(roi_hist,(20,1))
            #cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            # Use this to show the area you selected. Buggy.
            #cv2.namedWindow("selection")
            #cv2.imshow("selection", roi)

            #is_tracking = True

cv2.namedWindow("frame")
vc = cv2.VideoCapture(0) # This value can be different for different video inputs
cv2.setMouseCallback("frame", cv2click)

cv2.namedWindow("frame2")
vc2 = cv2.VideoCapture(1) # This value can be different for different video inputs
cv2.setMouseCallback("frame2", cv2click2)

reticle = cv2.imread("reticle.png")

def bincount_app(a): #returns dominant color
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

# This is only for overlays (the reticle)
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


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

'''
# We load the images
face_img = cv2.imread("lena.png", -1)
overlay_img = cv2.imread("overlay.png", -1)

result_1 = blend_non_transparent(face_img, overlay_img)
cv2.imwrite("merged.png", result_1)
'''

movetimer = 0

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    cv2.moveWindow("frame",0,0)
else:
    rval = False

if vc2.isOpened(): # try to get the first frame
    rval2, frame2 = vc2.read()
    cv2.moveWindow("frame2",500,0)
else:
    rval2 = False

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1 )

'''
#blobs
blobparams = cv2.SimpleBlobDetector_Params()
blobparams.filterByInertia = False
blobparams.filterByConvexity = False
blobparams.filterByColor = False
blobparams.filterByCircularity = False
blobparams.filterByArea = False
blobdetector = cv2.SimpleBlobDetector_create()
'''

prev10moves = []
prevcentroid = None
prevcentroid2 = None
centroid = None
centroid2 = None

while rval and rval2 and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) >= 1 and cv2.getWindowProperty("frame2", cv2.WND_PROP_VISIBLE) >= 1:
    #global mousex
    #global mousey
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()
    rval2, frame2 = vc2.read()

    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([110,0,0])
    upper_color = np.array([130,255,255])

    # Threshold the HSV image to get only intended colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    frame = res
    '''
    #draws a rectangle at the mouse position (debug)
    #cv2.rectangle(frame, (0, 0), (int(mousex), int(mousey) - 75), (255,0,0), 2)

    #moves the window to the top left corner of the screen
    #cv2.moveWindow("frame",0,0)

    if dragging:
        cv2.rectangle(frame, (beginRefPt[0], beginRefPt[1]), (mousex, mousey), (0,255,0), 2)

    if is_tracking:
        #incredibly laggy denoising
        #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        #huemask = cv2.inRange(hsv, np.asarray((huemin, 30.,22.)), np.asarray((huemax,255.,230.)))
        huemask = cv2.inRange(hsv, np.asarray((huemin,satmin,valmin)), np.asarray((huemax,satmax,valmax)))
        kernel = np.ones((8,8), np.uint8)
        huemaskdenoise = cv2.erode(huemask, kernel, iterations=1)
        huemaskdenoise = cv2.dilate(huemask, kernel, iterations=2)
        huemaskcolor = cv2.bitwise_and(frame,frame, mask=huemask)
        huemaskdenoisecolor = cv2.bitwise_and(frame,frame, mask=huemaskdenoise)
        dstcrop = cv2.bitwise_and(dst,dst, mask=huemaskdenoise)
        dstcrop = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        dstcrop = cv2.bitwise_and(dstcrop,dstcrop, mask=huemaskdenoise)
        #kernel = np.ones((8,8), np.uint8)
        #dstdenoise = cv2.erode(dst, kernel, iterations=1)
        #dstdenoise = cv2.dilate(dstdenoise, kernel, iterations=1)
        #dstdenoise = cv2.calcBackProject([huemaskdenoisecolor],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        '''the image is being written on from this point onward'''

        '''
        #blobs
        #huemask = (255-huemask)
        #huemask = grayscale
        #huemaskcolor = 255 - huemask
        maskv = cv2.cvtColor(huemaskdenoisecolor, cv2.COLOR_BGR2HSV)
        maskh, masks, maskv = cv2.split(huemaskdenoisecolor)
        maskv = cv2.bitwise_not(maskv)
        #huemaskcolor =
        maskv[maskv != 255] = 0
        kernel = np.ones((16,16), np.uint8)
        maskv = cv2.dilate(maskv, kernel, iterations=3)
        maskv = cv2.erode(maskv, kernel, iterations=2)
        #maskv = cv2.blur(maskv,(256,256))
        #maskv = cv2.dilate(maskv, kernel, iterations=2)
        blobpoints = blobdetector.detect(maskv)
        frame = cv2.drawKeypoints(maskv, blobpoints, np.array([]), (150,100,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        # Draw tracking box on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        frame = cv2.polylines(frame,[pts],True, 255,2)
        #cv2.imshow('img2',img2)
        centroid = (track_window[0] + int(track_window[2] / 2), track_window[1] + int(track_window[3] / 2))
        #centroid = (int((track_window[0] + track_window[2]) / 2), int((track_window[1] + track_window[3]) / 2))
        cv2.circle(frame, centroid, 3, (0, 0, 255), 3)

        #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #roi_hist = cv2.calcHist([hsv[track_window[0]:track_window[2], track_window[1]:track_window[3]]],[0],None,[30],[0,180])
        '''
        roi = crop_minAreaRect(frame, ret)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_mask = cv2.inRange(hsv_roi, np.asarray((huemin,satmin,valmin)), np.asarray((huemax,satmax,valmax)))
            #roi_mask = cv2.inRange(crop_minAreaRect(frame, ret), np.asarray((huemin,satmin,valmin)), np.asarray((huemax,satmax,valmax)))
            roi_hist = cv2.calcHist([hsv_roi],[0],roi_mask,[30],[0,180])
        except:
            print("failed")
        '''

        '''
        #this code makes the servo sentient
        movetimer += 1
        movecoefficient = 0.07 # 0.07 and 3 recommended; lower for more frequent motion, higher for less
        if movetimer == 3: # how often it should move
            movetimer = 0
            servohpos += (frame.shape[1] / 2 - centroid[0]) * movecoefficient
            if servohpos < 0:
                servohpos = 0
            elif servohpos > 180:
                servohpos = 180
            servoh.write(servohpos)
            servovpos -= (frame.shape[0] / 2 - centroid[1]) * movecoefficient
            if servovpos < 0:
                servovpos = 0
            elif servovpos > 180:
                servovpos = 180
            servov.write(servovpos)
        '''
        #debug to show other types of frames (hsv, etc.)
        #frame = huemaskcolor

        #k = cv2.waitKey(60) & 0xff
        #if k == 27:
        #    break
        #else:
        #    cv2.imwrite(chr(k)+".jpg",img2)

    #this adds the reticle overlay
    #frame = blend_non_transparent(frame, reticle)

    if dragging2:
        cv2.rectangle(frame2, (beginRefPt[0], beginRefPt[1]), (mousex, mousey), (0,255,0), 2)

    if is_tracking2:
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        dst2 = cv2.calcBackProject([hsv2],[0],roi_hist,[0,180],1)

        ret2, track_window2 = cv2.CamShift(dst2, track_window2, term_crit)

        # Draw tracking box on image 2
        pts2 = cv2.boxPoints(ret2)
        pts2 = np.int0(pts2)
        frame2 = cv2.polylines(frame2,[pts2],True, 255,2)
        #cv2.imshow('img2',img2)
        centroid2 = (track_window2[0] + int(track_window2[2] / 2), track_window2[1] + int(track_window2[3] / 2))
        #centroid = (int((track_window[0] + track_window[2]) / 2), int((track_window[1] + track_window[3]) / 2))
        cv2.circle(frame2, centroid2, 3, (0, 0, 255), 3)

        #take the arccos of the ratio of movement viewed by questionable over known
        if prevcentroid is not None:
            relativecentroid = (centroid[0]/frame.shape[1], centroid[1]/frame.shape[0])
            relativeprevcentroid = (prevcentroid[0]/frame.shape[1], prevcentroid[1]/frame.shape[0])
            centroidmove = (np.subtract(relativecentroid, relativeprevcentroid)).sum()/2
            #centroidmove[0] /= frame.shape[1]
            #centroidmove[1] /= frame.shape[0]
            relativecentroid2 = (centroid2[0]/frame2.shape[1], centroid2[1]/frame2.shape[0])
            relativeprevcentroid2 = (prevcentroid2[0]/frame2.shape[1], prevcentroid2[1]/frame2.shape[0])
            centroidmove2 = (np.subtract(relativecentroid2, relativeprevcentroid2)).sum()/2
            #centroidmove2[0] /= frame2.shape[1]
            #centroidmove2[1] /= frame2.shape[0]
            if centroidmove is not 0:
                #appending = centroidmove2 / centroidmove
                appending = np.arccos(centroidmove / centroidmove2)
                print("coeff is ", centroidmove / centroidmove2)
                if not math.isnan(appending):
                    prev10moves.append(appending)
                    #print(prev10moves)
                    if len(prev10moves) > 10:
                        prev10moves.pop(0)
                    theta = sum(prev10moves)/len(prev10moves)
                    print(theta)
            #except:
            #    None
        prevcentroid = centroid
        prevcentroid2 = centroid2

    cv2.imshow("frame",frame)
    cv2.imshow("frame2",frame2)

    key = cv2.waitKey(50)
    dirkey = cv2.waitKey(100)
    '''
    if key == 32: #pressed space
        print("You pressed space!")

        #if you have a light
        #light.write(0)

        #this is for click-to-move
        #servovpos -= ((frame.shape[0] / 2) / mousey) * 180
        #if servovpos < 0:
        #    servovpos = 0
        #elif servovpos > 180:
        #    servovpos = 180
        #print(servovpos)
    else:
        #if you have a light
        #light.write(1)
        None
    '''
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
    if key == 27: # exit on ESC
        servoh.write(0)
        servov.write(0)
        break
cv2.destroyAllWindows()
