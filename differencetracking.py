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
    servoh = board.get_pin('d:10:s')
    servov = board.get_pin('d:11:s')
    mosfet = board.get_pin('d:9:o')
    mosfet.write(0)
    primecountdown = 0

    servohpos = 65
    servovpos = 150

    servoh.write(servohpos)
    servov.write(servovpos)

    dragging = False

    beginRefPt = []
    endRefPt = []

    mousex = 0
    mousey = 0

    track_window = (0, 0, 0, 0)
    roi_hist = 0
    is_tracking = False
    window = None

    centroid = None

    movetimer = 0

    predictedcentroid = None
    predictionsdone = 0
    predictionaccuracysum = 0
    kalmansdone = 0
    kalmanaccuracysum = 0

    showpredictions = False
    showkalmans = False
    showvideohud = True

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
        global centroid
        global prevcentroid
        global centroidmotion
        global scopetimer
        global dt
        global X
        global P
        global A
        global Q
        global B
        global U
        global Y
        global H
        global R
        global kalmanframes
        global prevkalmans

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
            # if using selection viewing
            '''
            try:
                cv2.destroyWindow("selection")
            except:
                None
            '''
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            is_tracking = True
            #print("This is a cv2 unclick!")
            endRefPt = (x, y)
            #r, h, c, w = 250,90,400,125
            # set tracking window values
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
            if h == 0: # if height is 0, do not track
                h = 1
                is_tracking = False
            if w == 0: # if width is 0, do not track
                w = 1
                is_tracking = False
            #r, h, c, w = beginRefPt[1], endRefPt[1] - beginRefPt[1], beginRefPt[0], endRefPt[0] - beginRefPt[0]
            # sets the track window
            track_window = (c, r, w, h)

            # set up the ROI for tracking
            dummy, rawframe = vc.read()
            roi = rawframe[r:r+h, c:c+w]
            #print(roi.mean(axis=0).mean(axis=0))
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            #this is for average color
            #targethue = hsv_roi.mean(axis=0).mean(axis=0)[0]

            #this is for dominant color
            targethue = bincount_app(hsv_roi)[0]

            # sets min and max hues for mask
            huemin = targethue - 20
            if huemin < 0:
                huemin = 0
            huemax = targethue + 20
            if huemax > 180:
                huemax = 180
            #print("Hue selected is ", targethue)
            #values set to 60.,32. for fading out darkness
            # creates the mask of area in selected hues
            mask = cv2.inRange(hsv_roi, np.array((huemin, 30.,22.)), np.array((huemax,255.,230.)))
            # creates region of interest histogram using mask
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            # Use this to show the area you selected. Buggy.
            #cv2.namedWindow("selection")
            #cv2.imshow("selection", roi)

            #is_tracking = True
            centroid = None
            prevcentroid = None
            centroidmotion = (0, 0)
            scopetimer = 0

            # Kalman Filter stuff
            #time step of mobile movement
            dt = 1#0.1
            # Initialization of state matrices
            # values in Q and R seem to be tweakable to increase filtering but also rubberbanding
            X = np.array([[beginRefPt[0]+w/2], [beginRefPt[1]+h/2], [0.0], [0.0]])#np.array([[0.0], [0.0], [0.1], [0.1]]) # estimated state x, y, dx/dt, dy/dt
            P = np.diag((0.1, 0.1, 1, 1))#np.diag((0.01, 0.01, 0.01, 0.01)) # uncertainty of X
            A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) # values at each step (x = 1*original x + x*dt, y same, dx/dt = 1*original dx/dt, dy/dt same)
            Q = np.eye(X.shape[0])/10 # noise possibility (eg: wind, bumps)
            B = np.eye(X.shape[0]) # external control variables (eg: steering, braking, acceleration) control input model applied to U (these are direct, will increase position directly, not good)
            U = np.zeros((X.shape[0],1))#np.zeros((X.shape[0],1)) # control input / control vector (multiplies B)
            # Measurement matrices
            Y = np.array([[X[0,0]], [X[1,0]]])#np.array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] + abs(np.random.randn(1)[0])]]) # noisy position measurement
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # change (like velocity) multiples (1 for values that occur)
            R = np.eye(Y.shape[0])*1#np.zeros(Y.shape[0]) # uncertainty of H

            kalmanframes = 5
            #kalmansdone = 0
            #kalmanaccuracysum = 0
            #kalmanaccuracy = 0
            prevkalmans = np.zeros((kalmanframes, 2), dtype=int) # initialize array to contain past kalmans

    # sets up the frame and mouse events
    cv2.namedWindow("frame")
    vc2 = cv2.VideoCapture(0) # This value can be different for different video inputs
    vc = cv2.VideoCapture(1) # This value can be different for different video inputs
    cv2.setMouseCallback("frame", cv2click)

    reticle = cv2.imread("reticle.png", cv2.IMREAD_UNCHANGED)
    scope = cv2.imread("mlg-scope.png")

    def bincount_app(a): #returns dominant color
        a2D = a.reshape(-1,a.shape[-1])
        col_range = (256, 256, 256) # generically : a2D.max(0)+1
        a1D = np.ravel_multi_index(a2D.T, col_range)
        return np.unravel_index(np.bincount(a1D).argmax(), col_range)

    def add_weighted_arrays(arr1, w1, arr2, w2):
        return np.add(np.multiply(arr1, w1), np.multiply(arr2, w2))

    def get_image_pts(img):
        h, w = img.shape[:2]
        if w < h:
            pts = np.array([[0, h-1],[0, 0],[w-1, 0],[w-1, h-1]], dtype="float32")
        else:
            pts = np.array([[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]], dtype="float32")
        return pts

    def get_rotated_region(img, pts, w, h): # returns normalized version of rotated rectangle
        #window_width = track_window[2]
        #window_height = track_window[3]
        src_pts = pts.astype("float32")
        # corrdinate of the points in box points after the rectangle has been straightened
        if w < h:
            dst_pts = np.array([[0, h-1],[0, 0],[w-1, 0],[w-1, h-1]], dtype="float32")
        else:
            dst_pts = np.array([[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]], dtype="float32")
        # the perspective transformation matrix
        windowM = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle (w, h)
        return cv2.warpPerspective(img, windowM, (w, h))

    # This is only for overlays (like the reticle)
    def overlay_offset(background_img, overlay_img, point1, point2, alpha=1, blur=3, erode=3, transparency = False):
        point1 = np.asarray(point1, int)
        point2 = np.asarray(point2, int)

        w = abs(point1[0] - point2[0]) # width
        h = abs(point1[1] - point2[1]) # height
        tlpoint = [min(point1[0], point2[0]),min(point1[1], point2[1])] # top left corner
        brpoint = [max(point1[0], point2[0]),max(point1[1], point2[1])] # bottom right corner

        if (w == 0 or h == 0 or brpoint[0] <= 0 or brpoint[1] <= 0 or tlpoint[0] >= background_img.shape[1] or tlpoint[1] >= background_img.shape[0]):
            return background_img

        #resize overlay to fit background
        overlay_img = cv2.resize(overlay_img,(w,h))

        #if offscreen, cut to fit
        if tlpoint[0] < 0:
            overlay_img = overlay_img[:, -1 * tlpoint[0]:]
            tlpoint[0] = 0
        if tlpoint[1] < 0:
            overlay_img = overlay_img[-1 * tlpoint[1]:, :]
            tlpoint[1] = 0
        if brpoint[0] > background_img.shape[1]:# - 1:
            overlay_img = overlay_img[:,:(overlay_img.shape[1] - (brpoint[0] - background_img.shape[1]))]
            brpoint[0] = background_img.shape[1]# - 1
        if brpoint[1] > background_img.shape[0]:# - 1:
            overlay_img = overlay_img[:(overlay_img.shape[0] - (brpoint[1] - background_img.shape[0])),:]
            brpoint[1] = background_img.shape[0]# - 1

        # Let's find a mask covering all the non-black (foreground) pixels
        # NB: We need to do this on grayscale version of the image
        #gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        #overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]
        if not transparency:
            background_img[tlpoint[1]:brpoint[1], tlpoint[0]:brpoint[0]] = cv2.addWeighted(background_img[tlpoint[1]:brpoint[1], tlpoint[0]:brpoint[0]], 1-alpha, overlay_img, alpha, 0.0)
            return background_img

        if overlay_img.shape[2] == 4: # this part uses alpha as transparency
            alpha_or_grey_overlay = overlay_img[:,:,3]
        else: # this part uses black as transparency
            alpha_or_grey_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        overlay_mask = cv2.threshold(alpha_or_grey_overlay, 1, 255, cv2.THRESH_BINARY)[1]

        # Let's shrink and blur it a little to make the transitions smoother...
        overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
        overlay_mask = cv2.blur(overlay_mask, (blur, blur))

        # And the inverse mask, that covers all the black (background) pixels
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        background_part = (background_img[tlpoint[1]:brpoint[1], tlpoint[0]:brpoint[0]] * (1 / 255.0)) * (background_mask * (1 / 255.0))
        # combine and apply transparency
        overlay_part = cv2.addWeighted((overlay_img[:,:,:3] * (1 / 255.0)), alpha, (background_img[tlpoint[1]:brpoint[1], tlpoint[0]:brpoint[0]] * (1 / 255.0)), 1 - alpha, 0.0) * (overlay_mask * (1 / 255.0))

        background_img[tlpoint[1]:brpoint[1], tlpoint[0]:brpoint[0]] = cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0)#(background_part * 700.0 + overlay_part * 320.0) / 2 #cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0)# / 2

        return background_img
        # And finally just add them together, and rescale it back to an 8bit integer image
        #return np.uint8(cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0))

    # overlays image onto rotated region given the corner points of that region
    def overlay_to_rotated_region(background_img, overlay_img, dst_pts, alpha=1, blur=3, erode=3, transparency = True):
        src_pts = get_image_pts(overlay_img) # gets points of overlay image
        local_dst_pts = (dst_pts - np.amin(dst_pts, 0)).astype("float32") # localizes
        overlay_M = cv2.getPerspectiveTransform(src_pts, local_dst_pts)
        overlay = cv2.warpPerspective(overlay_img, overlay_M, tuple(np.amax(local_dst_pts, 0)))
        return overlay_offset(background_img, overlay, np.amin(dst_pts, 0), np.amax(dst_pts, 0), alpha, blur, erode, transparency)

    def kf_predict(X, P, A, Q, B, U):
        X = np.dot(A, X) + np.dot(B, U)
        P = np.dot(A, np.dot(P, A.T)) + Q
        return(X,P)

    def kf_update(X, P, Y, H, R):
        IM = np.dot(H, X)
        IS = R + np.dot(H, np.dot(P, H.T))
        K = np.dot(P, np.dot(H.T, np.linalg.inv(IS)))
        X = X + np.dot(K, (Y-IM))
        P = P - np.dot(K, np.dot(IS, K.T))
        LH = gauss_pdf(Y, IM, IS)
        return (X,P,K,IM,IS,LH)

    def gauss_pdf(X, M, S):
        if M.shape[1] == 1:
            DX = X - np.tile(M, X.shape[1])
            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape[0] * np.log(2 * math.pi) + 0.5 * np.log(np.linalg.det(S))
            P = np.exp(-E)
        elif X.shape[1] == 1:
            DX = tile(X, M.shape[1])- M
            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape[0] * np.log(2 * math.pi) + 0.5 * np.log(np.linalg.det(S))
            P = np.exp(-E)
        else:
            DX = X-M
            E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX))
            E = E + 0.5 * M.shape[0] * np.log(2 * math.pi) + 0.5 * np.log(np.linalg.det(S))
            P = np.exp(-E)
        return (P[0],E[0])

    '''
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
    '''
    # We load the images
    face_img = cv2.imread("lena.png", -1)
    overlay_img = cv2.imread("overlay.png", -1)

    result_1 = blend_non_transparent(face_img, overlay_img)
    cv2.imwrite("merged.png", result_1)
    '''

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

    k = 0 # number of clusters to search for
    clusters = None # array of cluster points
    clusterradii = None # array of cluster radii
    means = None # array of cluster means
    significance = None # array of cluster significances
    xstd = None # array of standard deviations of x values within clusters (used for rectangle width)
    ystd = None # array of standard deviations of y values within clusters (used for rectangle height)
    frameid = 0 # id of which frame to show
    scopetimer = 0

    while rval and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) >= 1:
        #print("-----------------")
        #global mousex
        #global mousey
        #cv2.imshow("preview", frame)
        # set up previous frames to help with difference tracking
        prevframe3 = prevframe2
        prevframe2 = prevframe
        prevframe = rawframe
        # read the video capture for frame
        rval, frame = vc.read()
        rval2, frame2 = vc2.read()
        rawframe = copy.deepcopy(frame)

        # set up arrays of previous values for comparison
        prevclusters = clusters
        prevclusterradii = clusterradii
        prevmeans = means
        prevxstd = xstd
        prevystd = ystd
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

        if dragging: # draw dragging rectangle
            cv2.rectangle(frame, (beginRefPt[0], beginRefPt[1]), (mousex, mousey), (0,255,0), 2)

        # k-means clustering via frame differences
        subframe = cv2.absdiff(prevframe, rawframe) # subtraction of previous frame and current frame
        subframe2 = cv2.absdiff(prevframe3, rawframe) # subtraction of older frame and current frame
        subhsv = cv2.cvtColor(subframe, cv2.COLOR_BGR2HSV) # hsv conversion
        subhsv2 = cv2.cvtColor(subframe2, cv2.COLOR_BGR2HSV) # hsv conversion
        subhsvmask = cv2.inRange(subhsv, np.array([0,0,20]), np.array([255,255,180])) # mask of only high enough values
        subhsvmask2 = cv2.inRange(subhsv2, np.array([0,0,20]), np.array([255,255,180])) # mask of only high enough values
        #subhsvmask = cv2.erode(subhsvmask, [1,1])
        subhsvmasks = cv2.bitwise_and(subhsvmask, subhsvmask2) # combines the masks
        subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmasks) # masks the subtracted frame
        #subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmask)
        #subframe = cv2.bitwise_and(subframe, subframe, mask = subhsvmask2)
        rawsubframe = subframe
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)) # circle kernel for erosion and dilation
        #kernel = np.ones((4,4),np.uint8) # square kernel for erosion and dilation
        subframe = cv2.erode(subframe, kernel, iterations = 2) # erode sound away
        bwsubframe = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY) # get black and white frame
        rawbwsubframe = cv2.cvtColor(rawsubframe, cv2.COLOR_BGR2GRAY) # no noise cancellation
        (thresh,bwsubframe) = cv2.threshold(bwsubframe,1,255,cv2.THRESH_BINARY) # snap values above 1 to 255
        (thresh,rawbwsubframe) = cv2.threshold(rawbwsubframe,1,255,cv2.THRESH_BINARY) # snap values above 1 to 255
        binarysub = bwsubframe / 255 # create binary-style image where 0 is black and 1 is white
        #print(binarysub)
        #print(np.argwhere(bwsubframe==255))
        #bgrbwsubframe = cv2.cvtColor(bwsubframe, cv2.COLOR_GRAY2BGR)
        #print(round(math.log10(np.sum(binarysub)+1)))
        #print(round(math.log10(np.sum(bwsubframe)+1)))

        samples = np.argwhere(binarysub==1) # get coordinate points of all white pixels
        samples32 = np.float32(samples)
        motionc = round(math.log10(np.sum(binarysub)+1)) # coefficient of how much motion is occurring
        #print the motion coefficient
        #print(motionc)
        k += 1
        #k = 3

        if len(samples32) < k:
            k = len(samples32)
        if k > 0 and len(samples) > 1:#len(samples) >= numclusters and numclusters > 0:
            # does the k-means, means are centers of clusters (y, x)
            compactness, cluster_labels, means = cv2.kmeans(data=samples32, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_PP_CENTERS) # Let OpenCV choose random centers for the clusters
            # makes lists of clusters
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
            clusters = [samples[cluster_labels.ravel()==i] for i in range(k)] # separates clusters into array
            #clusterradii = [np.var(clusters[i])/len(clusters[i]) for i in range(numclusters)]
            #calculates standard deviation to use as radius
            #clusterradii = [int(math.sqrt(np.var(i))) for i in clusters]
            #clusterradii = [int(np.std(i)) for i in clusters]
            stdc = 1.5 # coefficient to multiply the standard deviation by
            xstd = [int(np.std(i[:, 1]) * stdc) for i in clusters]
            ystd = [int(np.std(i[:, 0]) * stdc) for i in clusters]
            clusterradii = (np.add(ystd, xstd) / 2).astype(int) # gets array of cluster radii based on standard deviations

            mergeclusters = 0
            distc = 2 # coefficient of distance between two clusters for them to merge (multiply rectangle dimensions by this value. if the new rectangles overlap, they will merge)
            for i in range(k):
                for iteratorj in range(k - i - 1):
                    j = k - iteratorj - 1
                    ##if the circles are touching (distance less than sum of radii)
                    #if i!=j and np.linalg.norm(means[i] - means[j]) < clusterradii[i] + clusterradii[j]:#math.sqrt((means[i][0]-means[j][0])**2 + (means[i][1]-means[j][1])**2) < clusterradii[i] + clusterradii[j]:
                    # if the rectangles overlap
                    if i!=j and abs(means[i][1] - means[j][1]) < distc * (xstd[i] + xstd[j]) and abs(means[i][0] - means[j][0]) < distc * (ystd[i] + ystd[j]):
                        mergeclusters += 1
                        break
            if mergeclusters != 0 and k > mergeclusters: # if there are clusters to merge
                k -= mergeclusters # decrease k by number of clusters
                # does the k-means again, means are centers of clusters
                compactness, cluster_labels, means = cv2.kmeans(data=samples32, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_PP_CENTERS)   #Let OpenCV choose random centers for the clusters
                # makes array of clusters
                clusters = [samples[cluster_labels.ravel()==i] for i in range(k)]
                #clusterradii = [np.var(clusters[i])/len(clusters[i]) for i in range(numclusters)]
                #calculates standard deviation to use as radius
                #clusterradii = [int(math.sqrt(np.var(i))) for i in clusters]
                xstd = [int(np.std(i[:, 1]) * stdc) for i in clusters]
                ystd = [int(np.std(i[:, 0]) * stdc) for i in clusters]
                clusterradii = (np.add(ystd, xstd) / 2).astype(int)


            #draws the circles
            #[cv2.circle(frame, tuple(np.flip(means[i])), clusterradii[i], (0, 255, 200), 3) for i in range(k)]

            #if prevsignificance is None:
            #    prevsignificance = np.zeros(len(prevclusters))
            significance = np.zeros(len(clusters)) # prepares the significance array
            for i in range(k):
                #x,y,w,h = cv2.boundingRect(clusters[i])
                #cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 200))
                #cv2.rectangle(frame, (int(means[i][1] - (clusterradii[i]/2)), int(means[i][0] - (clusterradii[i]/2))), (int(means[i][1] + (clusterradii[i]/2)), int(means[i][0] + (clusterradii[i]/2))), (0, 255, 200))
                mindist = clusterradii[i]
                perror = 0.3 # error of difference in dimensions allowed to be considered the same cluster
                idealcluster = -1
                if prevclusters is not None:
                    for j in range(len(prevclusters)):
                        # if within cluster and in acceptable dimension difference range
                        #if np.linalg.norm(means[i] - prevmeans[j]) < mindist and (1 - perror) * clusterradii[i] < prevclusterradii[j] and prevclusterradii[j] < (1 + perror) * clusterradii[i]:
                        if np.linalg.norm(means[i] - prevmeans[j]) < mindist and (1 - perror) * xstd[i] < prevxstd[j] and prevxstd[j] < (1 + perror) * xstd[i] and (1 - perror) * ystd[i] < prevystd[j] and prevystd[j] < (1 + perror) * ystd[i]:
                            mindist = np.linalg.norm(means[i] - prevmeans[j])
                            idealcluster = j
                    if idealcluster != -1:
                        significance[i] = prevsignificance[j] + 1 # increment significance
                        # draw a line at the centroid, showing the motion of the cluster
                        cv2.line(frame, tuple(np.flip(means[i])), tuple(np.flip(prevmeans[idealcluster])), (0, 0, 255), 4)
                #cv2.circle(frame, tuple(np.flip(means[i])), clusterradii[i], (0, 255 - significance[i] * 20, 255), 3)
                # draw the rectangle with color based off significance
                cv2.rectangle(frame, (int(means[i][1] - xstd[i]), int(means[i][0] - ystd[i])), (int(means[i][1] + xstd[i]), int(means[i][0] + ystd[i])), (0, 255 - significance[i] * 10, 255), 3)
                #frame = overlay_offset(frame, reticle, (int(means[i][1] - xstd[i]), int(means[i][0] - ystd[i])), (int(means[i][1] + xstd[i]), int(means[i][0] + ystd[i])), 0.8)

                trackmin = 5 # value to start camshift on cluster
                pterrormin = 0.2 # percent tracking size error min (percent of cluster that tracking area is)
                pterrormax = 4.0 # percent tracking size error max
                tmindist = clusterradii[i] * 1.5 # minimum distance away that tracking area can be
                # don't start camshift again if already tracking the cluster
                iscamshiftcluster = True if (is_tracking is True and centroid is not None and np.linalg.norm(np.flip(means[i]) - centroid) < tmindist and pterrormin * xstd[i] < track_window[2] and track_window[2] < pterrormax * xstd[i] and pterrormin * ystd[i] < track_window[3] and track_window[3] < pterrormax * ystd[i]) else False
                #if is_tracking and centroid is not None:
                #    print(np.linalg.norm(means[i] - centroid), " and ", tmindist)

                '''
                if is_tracking and centroid is not None and np.linalg.norm(np.flip(means[i]) - centroid) >= tmindist and significance[i] >= trackmin and significance[i] == max(significance):
                    print("Failed due to location")
                    print(centroid)
                    print(np.flip(means[i]))
                    print(np.linalg.norm(np.flip(means[i]) - centroid), " and ", tmindist)
                if is_tracking and centroid is not None and np.linalg.norm(np.flip(means[i]) - centroid) < tmindist and significance[i] >= trackmin and significance[i] == max(significance) and not iscamshiftcluster:# and pterrormin * ystd[i] > track_window[3]:
                    print("failed because too big or small")
                    print(track_window[2], ">", pterrormax * xstd[i])
                    print(track_window[3], ">", pterrormax * ystd[i])
                    print(track_window[2], "<", pterrormin * xstd[i])
                    print(track_window[3], "<", pterrormin * ystd[i])
                '''

                #if iscamshiftcluster:
                #    print("ALREADY")
                if iscamshiftcluster is False and significance[i] >= trackmin and significance[i] == max(significance): # if most significant cluster and past required value
                    #print("BEGIN")
                    #try:
                    #    print((1 + pterrorshift - pterror) * xstd[i], "<", track_window[2])
                    #    print((1 + pterrorshift + pterror) * xstd[i], ">", track_window[2])
                    #except:
                    #    None

                    zoomc = 2 # amount to zoom into cluster to get region of interest; eliminates edges
                    c = int(means[i][1] - xstd[i] / zoomc)
                    r = int(means[i][0] - ystd[i] / zoomc)
                    w = int(xstd[i] * 2 / zoomc)
                    h = int(ystd[i] * 2 / zoomc)

                    track_window = (c, r, w, h)

                    roi = rawframe[r:r+h, c:c+w]
                    #roi = rawframe[int(means[i][0] - (ystd[i])):int(means[i][0] + (ystd[i])), int(means[i][1] - (xstd[i])):int(means[i][1] + (xstd[i]))]
                    #print((int(means[i][1] - (xstd[i]/2))), "and", int(means[i][1] + (xstd[i]/2)))
                    #print(roi.mean(axis=0).mean(axis=0))
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

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
                    #print("Hue selected is ", targethue)
                    #values set to 60.,32. for fading out darkness
                    mask = cv2.inRange(hsv_roi, np.array((huemin, 30.,22.)), np.array((huemax,255.,230.)))
                    #mask = cv2.inRange(hsv_roi, np.array((0, 30.,22.)), np.array((180,255.,230.)))
                    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    is_tracking = True
                    centroid = None
                    prevcentroid = None
                    centroidmotion = (0, 0)
                    scopetimer = 0

                    #predictionsdone = 0
                    #predictionaccuracysum = 0

                    # Kalman Filter stuff
                    #time step of mobile movement
                    dt = 1#0.1
                    # Initialization of state matrices
                    # values in Q and R seem to be tweakable to increase filtering but also rubberbanding
                    X = np.array([[means[i][1]], [means[i][0]], [0.0], [0.0]])#np.array([[0.0], [0.0], [0.1], [0.1]]) # estimated state x, y, dx/dt, dy/dt
                    P = np.diag((0.1, 0.1, 1, 1))#np.diag((0.01, 0.01, 0.01, 0.01)) # uncertainty of X
                    A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) # values at each step (x = 1*original x + x*dt, y same, dx/dt = 1*original dx/dt, dy/dt same)
                    Q = np.eye(X.shape[0])/10 # noise possibility (eg: wind, bumps)
                    B = np.eye(X.shape[0]) # external control variables (eg: steering, braking, acceleration) control input model applied to U (these are direct, will increase position directly, not good)
                    U = np.zeros((X.shape[0],1))#np.zeros((X.shape[0],1)) # control input / control vector (multiplies B)
                    # Measurement matrices
                    Y = np.array([[X[0,0]], [X[1,0]]])#np.array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] + abs(np.random.randn(1)[0])]]) # noisy position measurement
                    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # change (like velocity) multiples (1 for values that occur)
                    R = np.eye(Y.shape[0])*1#np.zeros(Y.shape[0]) # uncertainty of H

                    kalmanframes = 5
                    #kalmansdone = 0
                    #kalmanaccuracysum = 0
                    #kalmanaccuracy = 0
                    prevkalmans = np.zeros((kalmanframes, 2), dtype=int) # initialize array to contain past kalmans

        else:
            # get rid of old data
            clusters = None
            clusterradii = None
            means = None
            xstd = None
            ystd = None

        if is_tracking:
            hsv = cv2.cvtColor(rawframe, cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            #print(roi_hist)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            trackret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # track_window is x, y, w, h

            # get hist of tracked area
            #track_window = (c, r, w, h)
            #window = rawframe[track_window[1]:track_window[1]+track_window[3], track_window[0]:track_window[0]+track_window[2]]
            windowpts = cv2.boxPoints(trackret)
            windowpts = np.int0(windowpts)
            window = get_rotated_region(rawframe, windowpts, track_window[2], track_window[3])
            '''
            window_width = track_window[2]
            window_height = track_window[3]
            src_pts = pts.astype("float32")
            # corrdinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0, window_height-1],[0, 0],[window_width-1, 0],[window_width-1, window_height-1]], dtype="float32")

            # the perspective transformation matrix
            windowM = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle (w, h)
            window = cv2.warpPerspective(rawframe, windowM, (window_width, window_height))
            '''

            windowzoomc = 1.2
            window_height = window.shape[0]
            window_width = window.shape[1]
            cropped_window = window[int(window_height * (windowzoomc - 1) / (2 * windowzoomc)):int(window_height * (windowzoomc + 1) / (2 * windowzoomc)), int(window_width * (windowzoomc - 1) / (2 * windowzoomc)):int(window_width * (windowzoomc + 1) / (2 * windowzoomc))]
            hsv_window = cv2.cvtColor(cropped_window, cv2.COLOR_BGR2HSV)
            window_hist = cv2.calcHist([hsv_window],[0],None,[180],[0,180])
            cv2.normalize(window_hist,window_hist,0,255,cv2.NORM_MINMAX)

            # Draw tracking box on image
            #pts = cv2.boxPoints(trackret)
            #pts = np.int0(pts)
            frame = cv2.polylines(frame,[windowpts],True,255,2)

            #reticlepts = get_image_pts(reticle)
            #localwindowpts = (windowpts - np.amin(windowpts, 0)).astype("float32")
            #trackingreticleM = cv2.getPerspectiveTransform(reticlepts, localwindowpts)
            #trackingreticle = cv2.warpPerspective(reticle, trackingreticleM, tuple(np.amax(localwindowpts, 0)))
            #frame = cv2.rectangle(frame, tuple(np.amin(windowpts, 0)), tuple(np.amax(windowpts, 0)), (255, 0, 150), 3)
            frame = overlay_to_rotated_region(frame, reticle, windowpts, 0.7)
            #frame = overlay_offset(frame, trackingreticle, np.amin(windowpts, 0), np.amax(windowpts, 0), 0.7)

            if centroid is not None:
                prevcentroid = centroid
            centroid = tuple(np.asarray(trackret[0], int))#(track_window[0] + int(track_window[2] / 2), track_window[1] + int(track_window[3] / 2))
            #centroid = (int((track_window[0] + track_window[2]) / 2), int((track_window[1] + track_window[3]) / 2))
            cv2.circle(frame, centroid, 3, (0, 0, 255), 3)

            predictionmodernity = 0.3 # amount to incorporate new predictions
            predictionframes = 5 # how many frames ahead to predict
            if prevcentroid is not None:
                predictionsdone += 1
                centroidmotion = add_weighted_arrays(centroidmotion, 1 - predictionmodernity, np.subtract(centroid, prevcentroid), predictionmodernity) # incorporate new motion
                #np.add(np.multiply(centroidmotion, 1 - predictionmodernity), np.multiply(np.subtract(centroid, prevcentroid), predictionmodernity)))
                predictedcentroid = tuple((np.add(centroid, np.multiply(centroidmotion, predictionframes))).astype(int)) # multiply to get prediction
                if showpredictions:
                    cv2.circle(frame, predictedcentroid, 6, (255, 0, 255), 2)
                    cv2.line(frame, tuple(np.subtract(predictedcentroid, (10, 0))), tuple(np.add(predictedcentroid, (10, 0))), (255, 0, 255), 2)
                    cv2.line(frame, tuple(np.subtract(predictedcentroid, (0, 10))), tuple(np.add(predictedcentroid, (0, 10))), (255, 0, 255), 2)

                predictionaccuracysum += np.linalg.norm(centroid - prevpredictions[predictionframes - 1])
                predictionaccuracy = predictionaccuracysum / predictionsdone
                if (prevpredictions[predictionframes-1]!=[0, 0]).all(): # if the prediction frame has been reached
                    if showpredictions:
                        cv2.circle(frame, tuple(prevpredictions[predictionframes - 1]), 3, (0, 255, 0), 2)
                        frame = cv2.putText(frame, f"Linear accuracy: {round(predictionaccuracy, 3)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                prevpredictions = np.roll(prevpredictions, 1, axis=0) # move predictions down
                prevpredictions[0] = predictedcentroid # set first to new prediction
            else:
                prevpredictions = np.zeros((predictionframes, 2), dtype=int) # initialize array to contain past predictions

            # Kalman Filter
            kalmansdone += 1
            Y = np.array([[centroid[0]],[centroid[1]]])#np.array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] + abs(0.1 * randn(1)[0])]])
            #B = np.diag((centroid[0], centroid[1], centroid[0], centroid[1])) doesn't work at all
            (X, P) = kf_predict(X, P, A, Q, B, U)
            (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
            #Y = np.array([[centroid[0]],[centroid[1]]])#np.array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] + abs(0.1 * randn(1)[0])]])
            if showkalmans:
                cv2.circle(frame, (int(X[0,0]), int(X[1,0])), 6, (60, 180, 140), 5)

            #kalmanframes = 5
            (fX, fP) = kf_predict(X, P, A, Q, B, U)
            (fX, fP, fK, fIM, fIS, fLH) = kf_update(fX, fP, Y, H, R)
            for i in range(kalmanframes - 1):
                (fX, fP) = kf_predict(fX, fP, A, Q, B, U)
                (fX, fP, fK, fIM, fIS, fLH) = kf_update(fX, fP, fX[:2], H, R)
            if showkalmans:
                cv2.circle(frame, (int(fX[0,0]), int(fX[1,0])), 6, (50, 100, 100), 5)

            kalmanaccuracysum += np.linalg.norm(centroid - prevkalmans[kalmanframes - 1])
            kalmanaccuracy = kalmanaccuracysum / kalmansdone
            if (prevkalmans[kalmanframes-1]!=[0, 0]).all(): # if the kalman frame has been reached
                if showkalmans:
                    cv2.circle(frame, tuple(prevkalmans[kalmanframes - 1]), 5, (180, 100, 20), 4)
                    frame = cv2.putText(frame, f"Kalman accuracy: {round(kalmanaccuracy, 3)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            prevkalmans = np.roll(prevkalmans, 1, axis=0) # move kalmans down
            prevkalmans[0] = (int(fX[0,0]), int(fX[1,0])) # set first to new kalman

            '''
            scopetimer += 1
            scopespeed = 2
            scopeheight = 30
            frame = overlay_offset(frame, scope, (predictedcentroid[0] - 100, predictedcentroid[1] + int(scopeheight * abs(math.cos(scopetimer * scopespeed)))), (predictedcentroid[0] + 100, predictedcentroid[1] + 100 + int(scopeheight * abs(math.cos(scopetimer * scopespeed)))), 1)
            #if scopeheight * math.cos(scopetimer * scopespeed) < 0:
            #    frame = overlay_offset(frame, scope, (predictedcentroid[0] - 100, predictedcentroid[1] + int(scopeheight * math.cos(scopetimer * scopespeed))), (predictedcentroid[0] + 100, predictedcentroid[1] + 100 + int(scopeheight * math.cos(scopetimer * scopespeed))), 1)
            #else:
            #    frame = overlay_offset(frame, scope, (predictedcentroid[0] - 100, predictedcentroid[1]), (predictedcentroid[0] + 100, predictedcentroid[1] + 100), 1)
            '''
            # stop tracking if window hist is too different from roi hist
            #print(cv2.compareHist(roi_hist,window_hist,cv2.HISTCMP_CORREL))
            minhistdiff = 0.5 # minimum difference to keep tracking
            if cv2.compareHist(roi_hist,window_hist,cv2.HISTCMP_CORREL) < minhistdiff:
                is_tracking = False

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
        #frame = overlay_offset(frame, reticle, (0, 0), (frame.shape[1], frame.shape[0]), 0.2)

        '''
        subframe = cv2.subtract(prevframe, frame)
        alpha = 255.0 / 50
        beta = -50 * alpha
        subframe.convertTo(subframe, -1, alpha, beta)
        '''

        if frameid == 0: # toggle shown image on space bar press
            showframe = frame
        elif frameid == 1:
            showframe = bwsubframe
        elif frameid == 2:
            showframe = rawbwsubframe
        elif frameid == 3:
            if is_tracking:#window is not None:
                showframe = window
            else:
                frameid = 0
                showframe = frame
        elif frameid == 4:
            try:
                showframe = trackingreticle
            except:
                frameid = 0
                showframe = frame

        #cv2.imshow("frame",showframe)

        armed = False
        mobile = False
        ''' # motion based
        if motionc >= 3:
            if armed == True:
                mosfet.write(1)
            if primecountdown < 10:
                primecountdown += 1
        elif primecountdown > 0:
            if armed == True:
                mosfet.write(1)
            primecountdown -= 1
        else:
            mosfet.write(0)
        '''

        if showvideohud and is_tracking and centroid is not None:
            #frame = overlay_offset(frame, frame2, (0, 0), (140, 100), 0.7)
            #webcam is 640X480
            #frame[0:120,0:160] = cv2.resize(frame2, (160,120))
            videohudsize = 1
            #frame[centroid[1] - 240 * videohudsize:centroid[1] + 240 * videohudsize, centroid[0] - 320 * videohudsize:centroid[0] + 320 * videohudsize] = cv2.addWeighted(frame[centroid[1] - 240 * videohudsize:centroid[1] + 240 * videohudsize, centroid[0] - 320 * videohudsize:centroid[0] + 320 * videohudsize], 1, cv2.resize(frame2, videohudsize * (640,480)), 1, 0)
            frame = overlay_offset(frame, frame2, (centroid[0] - 320 * videohudsize, centroid[1] - 240 * videohudsize), (centroid[0] + 320 * videohudsize, centroid[1] + 240 * videohudsize), 0.7, transparency = False)

        # tracking based
        if is_tracking:
            if armed == True:
                mosfet.write(1)
            if mobile == True and predictedcentroid is not None:
                #movetimer += 1
                #movecoefficient = 20 # 0.07 and 3 recommended; lower for more frequent motion, higher for less
                #if movetimer == 3: # how often it should move
                #    movetimer = 0
                servohpos = 105 - predictedcentroid[0] / 15
                #servohpos = (frame.shape[1] / 2 - centroid[0]) * movecoefficient
                if servohpos < 0:
                    servohpos = 0
                elif servohpos > 180:
                    servohpos = 180
                servoh.write(servohpos)
                servovpos = 120 + predictedcentroid[1] / 15
                #servovpos = (frame.shape[0] / 2 - centroid[1]) * movecoefficient
                if servovpos < 0:
                    servovpos = 0
                elif servovpos > 180:
                    servovpos = 180
                servov.write(servovpos)
            primecountdown = 1
        else:
            primecountdown = 0
            mosfet.write(0)

        #if primecountdown > 0:
        #    frame = cv2.putText(frame, 'you are dieing', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("frame",showframe)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            frameid += 1
            if frameid >= 5:
                frameid = 0
        elif key == 107:
            showkalmans = not showkalmans
        elif key == 112:
            showpredictions = not showpredictions
        elif key == 118:
            showvideohud = not showvideohud
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
servov.write(140)
servoh.write(60)
cv2.destroyAllWindows()
print("Program execution complete")
