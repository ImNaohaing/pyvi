# coding=utf-8
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

im = np.array(Image.open('./images/empire.png').convert('L'))
H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
im2 = ndimage.affine_transform(im, H[:2, :2], offset=(H[0, 2], H[1, 2]))

plt.figure()
plt.gray()
plt.imshow(im2)
plt.show()

