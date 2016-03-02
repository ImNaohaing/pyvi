# coding=utf-8
from pyvision import harris
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pyvision import sift

imname1 = './images/compare1.jpg'
imsiftname1 = './data/tmpim1.sift'
im1 = np.array(Image.open(imname1).convert('L'))
sift.process_image(imname1, imsiftname1)
l1, d1 = sift.read_features_from_file(imsiftname1)
# plt.figure()
# plt.gray()
# sift.plot_features(im1, l1, circle=True)
# plt.show()
os.remove(os.path.abspath(imsiftname1))

imname2 = './images/compare2.jpg'
imsiftname2 = './data/tmpim2.sift'
im2 = np.array(Image.open(imname2).convert('L'))
sift.process_image(imname2, imsiftname2)
l2, d2 = sift.read_features_from_file(imsiftname2)
# plt.figure()
# plt.gray()
# sift.plot_features(im2, l2, circle=True)
# plt.show()
os.remove(os.path.abspath(imsiftname2))
print 'starting matching'
matchscores = sift.match_twosided(d1, d2)
print matchscores
plt.figure()
plt.gray()
harris.plot_matches3(im1, im2, l1, l2, matchscores[:, 0])
plt.show()

