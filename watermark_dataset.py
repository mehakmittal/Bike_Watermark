
# import the necessary packages

import numpy as np
#import urllib
from imutils import paths
import argparse
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--watermark", required=True,
    help="path to watermark image (assumed to be transparent PNG)")
ap.add_argument("-i", "--input", required=True,
	help="path to the input directory of images")
ap.add_argument("-o", "--output", required=True,
	help="path to the output directory")
ap.add_argument("-a", "--alpha", type=float, default=0.13,
	help="alpha transparency of the overlay (smaller is more transparent)")
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())

# load the watermark image, making sure we retain the 4th channel
# which contains the alpha transparency
watermark = cv2.imread(args["watermark"], cv2.IMREAD_UNCHANGED)
(wH, wW) = watermark.shape[:2]

# split the watermark into its respective Blue, Green, Red, and
# Alpha channels; then take the bitwise AND between all channels
# and the Alpha channels to construct the actaul watermark
# NOTE: I'm not sure why we have to do this, but if we don't,
# pixels are marked as opaque when they shouldn't be
if args["correct"] > 0:
    (B, G, R, A) = cv2.split(watermark)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    watermark = cv2.merge([B, G, R, A])


# METHOD #1: OpenCV, NumPy, and urllib
# def url_to_image(url):
# 	# download the image, convert it to a NumPy array, and then read
# 	# it into OpenCV format
# 	resp = urllib.urlopen(url)
# 	image = np.asarray(bytearray(resp.read()), dtype="uint8")
# 	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# 	# return the image
# 	return image

# urls=["https://dt2czl74lk540.cloudfront.net/cw8F9mUXIYNoIGR8a_XZFndYBNw=/fit-in/780x446/http://assets.credr.com/bike_images/MH04HB0710_4.JPG"
# ]

# loop over the image URLs
for imagePath in paths.list_images(args["input"]):
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

    # construct an overlay that is the same size as the input
    # image, (using an extra dimension for the alpha transparency),
    # then add the watermark to the overlay in the bottom-right
    # corner
    overlay = np.zeros((h, w, 4), dtype="uint8")
    #overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark
    overlay[h - wH:h, w / 2 - wW / 2:w / 2 + wW / 2] = watermark

    # blend the two images together using transparent overlays
    output = image.copy()
    cv2.addWeighted(overlay, args["alpha"], output, 1.0, 0, output)

    # write the output image to disk
    filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
    p = os.path.sep.join((args["output"], filename))
    cv2.imwrite(p, output)
    # cv2.imshow('output',output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
