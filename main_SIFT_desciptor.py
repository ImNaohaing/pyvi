# coding=utf-8
from pyvision import harris
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from pyvision import sift

imname = './images/shuicaili.jpg'
imsiftname = './data/tmpim.sift'
im1 = np.array(Image.open(imname).convert('L'))
sift.process_image(imname, imsiftname)
l1, d1 = sift.read_features_from_file(imsiftname)

plt.figure()
plt.gray()
sift.plot_features(im1, l1, circle=True)
plt.show()

os.remove(os.path.abspath(imsiftname))