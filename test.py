import cv2
import numpy as np
import skimage.io
from skimage import exposure, img_as_ubyte

import skimage
import numpy as np
from pathlib import Path
import skimage.io
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage import data
from skimage import measure
from skimage.exposure import histogram, equalize_adapthist
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
# img = cv2.imread(r'd:\test.png', 0);
# # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
# #             cv2.THRESH_BINARY,41,10)
# # vis = img.copy()
#
# # img = cv2.GaussianBlur(img,(5,5),0)
# # ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# kernel = np.ones((9,9),np.uint8)
# # kernel2 = np.ones((3,3),np.uint8)
# img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
# # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
# img = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('img', th3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def find_cells(frame, fgbg, split=0.96, disk_r=7):
    if type(frame) != np.uint8:
        cells = img_as_ubyte(exposure.rescale_intensity(frame))
    else:
        cells = frame

    # hist, hist_centers = histogram(cells)
    # sums = np.cumsum(hist / hist.max())

    # thr_lo = hist_centers[np.argmax(sums > (split * sum(hist / hist.max())))]
    # thr_hi = thr_lo + 1

    # markers = np.zeros_like(cells)
    # markers[cells < thr_lo] = 1
    # markers[cells > thr_hi] = 2
    markers = fgbg.apply(frame) + 1
    elevation_map = sobel(cells)
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    selem = disk(disk_r)
    morph = opening(segmentation, selem)

    #     # Find contours at a constant value of 0.8
    #     contours = measure.find_contours(morph, 0.8)

    labeled_cells, _ = ndi.label(morph)
    # intensity image is added to provide data for further statistics computation
    regions = measure.regionprops(labeled_cells, intensity_image=frame)

    return regions

# i = 0
# while(1):
#   fgbg = cv2.createBackgroundSubtractorMOG2()
#   while(i < 70):
#     image_split = image[i, 1, :, :]
#     frame = img_as_ubyte(exposure.rescale_intensity(image_split))
#     print(i)
#     fgmask = fgbg.apply(frame)
#
#     cv2.imshow('frame', fgmask)
#     i += 1
#
#
#     k = cv2.waitKey(0) & 0xff
#     if k == 27:
#         break

image = skimage.io.imread(r"g:\Můj disk\examples\R2D2-40x-1.tif")
fgbg = cv2.createBackgroundSubtractorMOG2()
frames = []
frames_c = len(image) - 1
for idx, frame in enumerate(image):
            regions = find_cells(frame[1, :, :],fgbg)
            frames.append(regions)
            print('Frame ' + str(idx) + '/' + str(frames_c) + ' done. Found ' + str(len(regions)) + ' cells.')

cv2.destroyAllWindows()


