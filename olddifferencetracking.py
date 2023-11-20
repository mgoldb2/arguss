from pyfirmata import Arduino, util
import time
import math
import numpy as np
import cv2
import copy

try:
    print("Running...")
    #this value is 14101 for left USB port and 14201 for right, in my experience
    board = Arduino('/dev/cu.usbmodem14101')
    it = util.Iterator(board)
    it.start()

    print("Arduino ready")

    #light (unnecessary)
    #light = board.get_pin('d:13:o')
    #servoh = board.get_pin('d:9:s')
    #servov = board.get_pin('d:10:s')
    mosfet = board.get_pin('d:9:o')
    mosfet.write(0)
    primecountdown = 0

    #servohpos = 0
    #servovpos = 0

    dragging = False

    beginRefPt = []
    endRefPt = []

    mousex = 0
    mousey = 0

    track_window = (0, 0, 0, 0)
    roi_hist = 0
    is_tracking = False

    def cv2click(event, x, y, flags, param):
        # grab references to the global variables
        global beginRefPt
        global endRefPt
        global servohpos
        global servovpos
        global is_tracking
        global roi_hist
        global track_window
        global dragging
        global mousex
        global mousey

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

            # set up the ROI for tracking
            dummy, rawframe = vc.read()
            roi = rawframe[r:r+h, c:c+w]
            #print(roi.mean(axis=0).mean(axis=0))
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            #this is for average color
            #targethue = hsv_roi.mean(axis=0).mean(axis=0)[0]

            #this is for dominant color
            targethue = bincount_app(hsv_roi)[0]

            huemin = targethue - 20
            if huemin < 0:
                huemin = 0
            huemax = targethue + 20
            if huemax > 180:
                huemax = 180
            print("Hue selected is ", targethue)
            #values set to 60.,32. for fading out darkness
            mask = cv2.inRange(hsv_roi, np.array((huemin, 30.,22.)), np.array((huemax,255.,230.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            # Use this to show the area you selected. Buggy.
            #cv2.namedWindow("selection")
            #cv2.imshow("selection", roi)

            #is_tracking = True

    cv2.namedWindow("frame")
    vc = cv2.VideoCapture(0) # This value can be different for different video inputs
    cv2.setMouseCallback("frame", cv2click)

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
        rawframe = copy.deepcopy(frame)
        prevframe = rawframe
        prevframe2 = prevframe
        cv2.moveWindow("frame",0,0)
    else:
        rval = False

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1 )

    k = 0
    clusters = None
    clusterradii = None
    means = None
    significance = None
    frameid = 0

    while rval and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) >= 1:
        #print("-----------------")
        #global mousex
        #global mousey
        #cv2.imshow("preview", frame)
        prevframe3 = prevframe2
        prevframe2 = prevframe
        prevframe = rawframe
        rval, frame = vc.read()
        rawframe = copy.deepcopy(frame)

        prevclusters = clusters
        prevclusterradii = clusterradii
        prevmeans = means

        prevsignificance = significance

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

        #k-means clustering via frame differences
        subframe = cv2.absdiff(prevframe, rawframe)
        subframe2 = cv2.absdiff(prevframe3, rawframe)
        subhsv = cv2.cvtColor(subframe, cv2.COLOR_BGR2HSV)
        subhsv2 = cv2.cvtColor(subframe2, cv2.COLOR_BGR2HSV)
        subhsvmask = cv2.inRange(subhsv, np.array([0,0,20]), np.array([255,255,180]))
        subhsvmask2 = cv2.inRange(subhsv2, np.array([0,0,20]), np.array([255,255,180]))
        #subhsvmask = cv2.erode(subhsvmask, [1,1])
        subhsvmasks = cv2.bitwise_and(subhsvmask, subhsvmask2)
        subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmasks)
        #subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmask)
        #subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmask2)
        rawsubframe = subframe
        kernel = np.ones((4,4),np.uint8)
        subframe = cv2.erode(subframe, kernel, iterations = 2)
        bwsubframe = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
        rawbwsubframe = cv2.cvtColor(rawsubframe, cv2.COLOR_BGR2GRAY)
        (thresh,bwsubframe) = cv2.threshold(bwsubframe,1,255,cv2.THRESH_BINARY)
        (thresh,rawbwsubframe) = cv2.threshold(rawbwsubframe,1,255,cv2.THRESH_BINARY)
        binarysub = bwsubframe / 255
        #print(binarysub)
        #print(np.argwhere(bwsubframe==255))
        bgrbwsubframe = cv2.cvtColor(bwsubframe, cv2.COLOR_GRAY2BGR)
        #print(round(math.log10(np.sum(binarysub)+1)))
        #print(round(math.log10(np.sum(bwsubframe)+1)))

        samples = np.argwhere(binarysub==1)
        samples32 = np.float32(samples)
        motionc = round(math.log10(np.sum(binarysub)+1))
        #print the motion coefficient
        #print(motionc)
        k += 1
        #k = 3

        if len(samples32) < k:
            k = len(samples32)
        if k > 0 and len(samples) > 1:#len(samples) >= numclusters and numclusters > 0:
            #does the k-means, means are centers of clusters
            compactness, cluster_labels, means = cv2.kmeans(data=samples32, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_PP_CENTERS)   #Let OpenCV choose random centers for the clusters
            #makes lists of clusters
            '''
            for i in range(k):
                if len(means[i]) == 1:
                    means[i] = [0.0, 0.0]
                    means = np.delete(means, i)
            '''

            #if k == 1:
            #    clusters = samples[0]
            #else:
            #    clusters = [samples[cluster_labels.ravel()==i] for i in range(k)]
            clusters = [samples[cluster_labels.ravel()==i] for i in range(k)]
            #clusterradii = [np.var(clusters[i])/len(clusters[i]) for i in range(numclusters)]
            #calculates standard deviation to use as radius
            clusterradii = [int(math.sqrt(np.var(i))) for i in clusters]
            mergeclusters = 0
            for i in range(k):
                for iteratorj in range(k - i - 1):
                    j = k - iteratorj - 1
                    #if the circles are touching (distance less than sum of radii)
                    if i!=j and np.linalg.norm(means[i] - means[j]) < clusterradii[i] + clusterradii[j]:#math.sqrt((means[i][0]-means[j][0])**2 + (means[i][1]-means[j][1])**2) < clusterradii[i] + clusterradii[j]:
                        mergeclusters += 1
                        break
            if mergeclusters != 0 and k > mergeclusters:
                k -= mergeclusters
                #does the k-means, means are centers of clusters
                compactness, cluster_labels, means = cv2.kmeans(data=samples32, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_PP_CENTERS)   #Let OpenCV choose random centers for the clusters
                #makes lists of clusters
                clusters = [samples[cluster_labels.ravel()==i] for i in range(k)]
                #clusterradii = [np.var(clusters[i])/len(clusters[i]) for i in range(numclusters)]
                #calculates standard deviation to use as radius
                clusterradii = [int(math.sqrt(np.var(i))) for i in clusters]
            '''
            #mergingclusters = np.array([[]])
            mergingclusters = np.empty((0, 1),int) #this limits the dimensions to 1 and is generally bad
            #mergingclusters = np.array([[1,1],[3,2]])
            for i in range(numclusters):
                for iteratorj in range(numclusters - i):
                    j = numclusters - iteratorj - 1
                    #if the circles are touching (distance less than sum of radii)
                    #and [j, i] not in mergingclusters
                    if i!=j and math.sqrt((means[i][0]-means[j][0])**2 + (means[i][1]-means[j][1])**2) < clusterradii[i] + clusterradii[j]:
                        newcluster = np.array([i,j])
                        mergingclusters = np.append(mergingclusters,np.array([[i,j]]),axis=0)
                        #mergingclusters = np.concatenate((mergingclusters,[[i, j]]),axis=1)
            #finalclusters = copy.deepcopy(clusters)
            fmeans = []
            #fclusters = np.array([[]])
            fclusters = np.empty([1, 1])
            fclusterradii = []
            for i in range(numclusters):
                if mergingclusters[0] is not None and i in mergingclusters[:,0]: #this cluster will undergo a merge
                    mergeindexes = np.where(mergingclusters[:,0] == i)[0]
                    for mergeindex in mergeindexes:
                        j = int(mergingclusters[mergeindex][1])
                        #overlap distance is radii sum - distance apart
                        #radius is half of distance apart + sum of radii
                        newradius = int(clusterradii[i] + clusterradii[j] + math.sqrt((means[i][0]-means[j][0])**2 + (means[i][1]-means[j][1])**2) / 2)
                        #mean is weighted average-type equation based on radii
                        newmean = ((means[i] * clusterradii[i] + means[j] * clusterradii[j]) / (clusterradii[i] + clusterradii[j])).astype(int)
                        fclusterradii.append(newradius)
                        fmeans.append(newmean)
                        mergedcluster = np.append(clusters[i],clusters[j],axis=0)
                        fclusters = np.append(fclusters,mergedcluster,axis=0)
                elif len(mergingclusters) <= 0 or not i in mergingclusters[:,1]: #this cluster will not undergo a merge
                    fmeans.append(means[i])
                    fclusters = np.append(fclusters,clusters[i],axis=0)
                    fclusterradii.append(clusterradii[i])
            fnumclusters = len(fmeans)
            '''

            #draws the circles
            #[cv2.circle(frame, tuple(np.flip(means[i])), clusterradii[i], (0, 255, 200), 3) for i in range(k)]

            #if prevsignificance is None:
            #    prevsignificance = np.zeros(len(prevclusters))
            significance = np.zeros(len(clusters))
            for i in range(k):
                #x,y,w,h = cv2.boundingRect(clusters[i])
                #cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 200))
                #cv2.rectangle(frame, (int(means[i][1] - (clusterradii[i]/2)), int(means[i][0] - (clusterradii[i]/2))), (int(means[i][1] + (clusterradii[i]/2)), int(means[i][0] + (clusterradii[i]/2))), (0, 255, 200))
                mindist = clusterradii[i]
                perror = 0.2
                idealcluster = -1
                if prevclusters is not None:
                    for j in range(len(prevclusters)):
                        #if within cluster and in acceptable size difference range
                        if np.linalg.norm(means[i] - prevmeans[j]) < mindist and (1 - perror) * clusterradii[i] < prevclusterradii[j] and prevclusterradii[j] < (1 + perror) * clusterradii[i]:
                            mindist = np.linalg.norm(means[i] - prevmeans[j])
                            idealcluster = j
                    if idealcluster != -1:
                        significance[i] = prevsignificance[j] + 1
                        cv2.line(frame, tuple(np.flip(means[i])), tuple(np.flip(prevmeans[idealcluster])), (0, 0, 255), 4)
                cv2.circle(frame, tuple(np.flip(means[i])), clusterradii[i], (0, 255 - significance[i] * 20, 255), 3)
        else:
            clusters = None
            clusterradii = None
            means = None

        if is_tracking:
            hsv = cv2.cvtColor(rawframe, cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            #print(roi_hist)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            frame = cv2.polylines(frame,[pts],True, 255,2)
            #cv2.imshow('img2',img2)
            centroid = (track_window[0] + int(track_window[2] / 2), track_window[1] + int(track_window[3] / 2))
            #centroid = (int((track_window[0] + track_window[2]) / 2), int((track_window[1] + track_window[3]) / 2))
            cv2.circle(frame, centroid, 3, (0, 0, 255), 3)

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
            #frame = hsv

        '''
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
        elif key == 32:
            frameid += 1
            if frameid > 2:
                frameid = 0
        else:
            None
            #cv2.imwrite(chr(k)+".jpg",img2)
        '''
        #this adds the reticle overlay
        #frame = blend_non_transparent(frame, reticle)

        '''
        subframe = cv2.subtract(prevframe, frame)
        alpha = 255.0 / 50
        beta = -50 * alpha
        subframe.convertTo(subframe, -1, alpha, beta)
        '''
        '''
        if is_tracking:
            showframe = frame
        else:
            showframe = frame
        '''
        if frameid == 0:
            showframe = frame
        elif frameid == 1:
            showframe = bwsubframe
        elif frameid == 2:
            showframe = rawbwsubframe

        cv2.imshow("frame",showframe)

        if motionc >= 3:
            mosfet.write(1)
            if primecountdown < 10:
                primecountdown += 1
        elif primecountdown > 0:
            mosfet.write(1)
            primecountdown -= 1
        else:
            mosfet.write(0)


        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            frameid += 1
            if frameid >= 3:
                frameid = 0
        #dirkey = cv2.waitKey(100)
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
        '''
except KeyboardInterrupt:
    None
mosfet.write(0)
cv2.destroyAllWindows()
print("Program execution complete")
