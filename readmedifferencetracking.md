## This is the readme for **differencetracking.py** ##

Press <kbd>Space</kbd> while running to toggle views.

Press <kbd>V</kbd> to show the second camera's view.

Press <kbd>P</kbd> to show linear predictions.

Press <kbd>K</kbd> to show Kalman predictions.

Frames are subtracted to find what changed between them.
This is thresholded to black and white.

<img src="https://github.com/mgoldb2/arguss/blob/master/images/difference.png" width="300">

This is then denoised and turned into a binary-style image.

<img src="https://github.com/mgoldb2/arguss/blob/master/images/denoiseddifference.png" width="300">

K-means clustering is done on this image. The clusters are drawn as rectangles.

<img src="https://github.com/mgoldb2/arguss/blob/master/images/cluster.png" width="300">

Each cluster is evaluated to determine whether it existed in the previous frame.
Each frame a cluster exists, its significance increases.
The more significant a cluster is, the redder it will appear.
Once a cluster reaches a certain significance, camshift tracking will begin on it.
Camshift can also be manually started by clicking and dragging over an area on the screen.

<img src="https://github.com/mgoldb2/arguss/blob/master/images/camshift.png" width="300">

If the area inside the camshift box becomes too different from what it is looking for, it will stop tracking.

- - - -

### Info on Data Storage ###

Here are some ways data is stored in the CamShift algorithm:

* *trackret* stores the rotated rectangle's properties, ((centroidx, centroidy), (w, h), angle)
* *track_window* stores the rotated rectangle's properties, (x, y, w, h)
* *windowpts* is the four corners of the rotated rectangle
* *localwindowpts* is windowpts moved to (0, 0), has smallest possible dimensions

Note that some point values, like kmeans means, are stored as (y, x) rather than (x, y).

- - - -

Kalman Filter code adapted from [this paper](https://arxiv.org/pdf/1204.0375.pdf)
