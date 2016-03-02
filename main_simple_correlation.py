# coding=utf-8
from pyvision import harris
import numpy as np
from PIL import Image

from pyvision import harris
import matplotlib.pyplot as plt

# im = np.array(Image.open('compare1.jpg').convert('L'))
# harrisim = harris.compute_harris_response(im)
# filtered_coords = harris.get_harris_points(harrisim, 10, 0.1)
# harris.plot_harris_points(im, filtered_coords)
# print len(filtered_coords)
# # --------------------------------------------------------------



im1 = np.array(Image.open('./images/th.jpeg').convert('L'))
im2 = np.array(Image.open('./images/th.jpeg').convert('L'))

wid = 10
harrisim = harris.compute_harris_response(im1,5)
filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
d1 = harris.get_descriptors(im1,filtered_coords1,wid)
harrisim = harris.compute_harris_response(im2,5)
filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
d2 = harris.get_descriptors(im2,filtered_coords2,wid)

print 'starting matching'
matches = harris.match_twosided(d1,d2)
plt.figure()
plt.gray()
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
plt.show()


# plt.figure()
# plt.gray()
# im3 = harris.appendimages(im1, im2)
# if True:
#     im3 = np.vstack((im3, im3))
#     plt.imshow(im3)
#     cols1 = im1.shape[1]
#     plt.plot([p[1] for p in filtered_coords1], [p[0] for p in filtered_coords1], 'o')
#
#     plt.plot([p[1]+cols1 for p in filtered_coords2], [p[0] for p in filtered_coords2], 'o')
# plt.show()