# coding=utf-8
import numpy as np

from pyvision import sift, homography, warp
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import os

imgnum_range = range(2, 4)
featname = ['./data/Sweden' + str(i) + '.sift' for i in imgnum_range]
imname = ['./images/Sweden' + str(i) + '.png' for i in imgnum_range]

ims = []
l = []
d = []
matches = []
for i, fn in enumerate(featname):
    ims.append(np.array(Image.open(imname[i]).convert('L')))
    sift.process_image(imname[i], fn)
    ldtuple = sift.read_features_from_file(fn)
    l.append(ldtuple[0])
    d.append(ldtuple[1])
    os.remove(os.path.abspath(fn))

for i in range(imgnum_range[-1] - imgnum_range[0]):
    print('start matching two sides between %s and %s' % (imname[i], imname[i + 1]))
    matches.append(sift.match_twosided(d[i], d[i+1]))

matches
plt.figure()
plt.gray()
sift.plot_matches_show3(ims, l, matches)
plt.show()

#
# # function to convert the matches to hom. points
# def convert_points(j):
#     ndx = matches[j].nonzero()[0]
#     fp = homography.make_homog(l[j + 1][ndx, :2].T)
#     ndx2 = [int(matches[j][i]) for i in ndx]
#     tp = homography.make_homog(l[j][ndx2, :2].T)
#     return fp, tp
#
#
# # estimate the homographies
# model = homography.RansacModel()
# fp, tp = convert_points(1)
# H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
# fp, tp = convert_points(0)
# H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
# tp, fp = convert_points(2)  # NB: reverse order
# H_32 = homography.H_from_ransac(fp, tp, model)[0]  # im 3 to 2
# tp, fp = convert_points(3)  # NB: reverse order
# H_43 = homography.H_from_ransac(fp, tp, model)[0]  # im 4 to 3
#
#
# def panorama(H, fromim, toim, padding=2400, delta=2400):
#     """ Create horizontal panorama by blending two images
#     using a homography H (preferably estimated using RANSAC).
#     The result is an image with the same height as toim. 'padding'
#     specifies number of fill pixels and 'delta' additional translation. """
#     # check if images are grayscale or color
#     is_color = len(fromim.shape) == 3
#
#     # homography transformation for geometric_transform()
#     def transf(p):
#         p2 = np.dot(H, [p[0], p[1], 1])
#         return (p2[0] / p2[2], p2[1] / p2[2])
#
#     if H[1, 2] < 0:  # fromim is to the right
#         print 'warp - right'
#         # transform fromim
#         if is_color:
#             # pad the destination image with zeros to the right
#             toim_t = np.hstack((toim, np.zeros((toim.shape[0], padding, 3))))
#             fromim_t = np.zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
#             for col in range(3):
#                 fromim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
#                                                                   transf, (toim.shape[0], toim.shape[1] + padding))
#         else:
#             # pad the destination image with zeros to the right
#             toim_t = np.hstack((toim, np.zeros((toim.shape[0], padding))))
#             fromim_t = ndimage.geometric_transform(fromim, transf,
#                                                    (toim.shape[0], toim.shape[1] + padding))
#     else:
#         print 'warp - left'
#         # add translation to compensate for padding to the left
#         H_delta = np.array([[1, 0, 0], [0, 1, -delta], [0, 0, 1]])
#         H = np.dot(H, H_delta)
#         # transform fromim
#         if is_color:
#             # pad the destination image with zeros to the left
#             toim_t = np.hstack((np.zeros((toim.shape[0], padding, 3)), toim))
#             fromim_t = np.zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
#             for col in range(3):
#                 fromim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
#                                                                   transf, (toim.shape[0], toim.shape[1] + padding))
#         else:
#             # pad the destination image with zeros to the left
#             toim_t = np.hstack((np.zeros((toim.shape[0], padding)), toim))
#             fromim_t = ndimage.geometric_transform(fromim,
#                                                    transf, (toim.shape[0], toim.shape[1] + padding))
#             # blend and return (put fromim above toim)
#     if is_color:
#         # all non black pixels
#         alpha = ((fromim_t[:, :, 0] * fromim_t[:, :, 1] * fromim_t[:, :, 2]) > 0)
#         for col in range(3):
#             toim_t[:, :, col] = fromim_t[:, :, col] * alpha + toim_t[:, :, col] * (1 - alpha)
#     else:
#         alpha = (fromim_t > 0)
#         toim_t = fromim_t * alpha + toim_t * (1 - alpha)
#     return toim_t
#
# # warp the images
# delta = 2000 # for padding and translation
# im1 = np.array(Image.open(imname[1]))
# im2 = np.array(Image.open(imname[2]))
# im_12 = warp.panorama(H_12,im1,im2,delta,delta)
# im1 = np.array(Image.open(imname[0]))
# im_02 = warp.panorama(np.dot(H_12,H_01),im1,im_12,delta,delta)
# im1 = np.array(Image.open(imname[3]))
# im_32 = warp.panorama(H_32,im1,im_02,delta,delta)
# im1 = np.array(Image.open(imname[j+1]))
# im_42 = warp.panorama(np.dot(H_32,H_43),im1,im_32,delta,2*delta)
