# coding=utf-8
from pyvision import warp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pylab

# example of affine warp of im1 onto im2
im1 = np.array(Image.open('./images/empire.png').convert('L'))
im2 = np.array(Image.open('./images/compare1.jpg').convert('L'))
# set to points
plt.figure()
plt.gray()
plt.imshow(im2)
plt.axis('equal')
plt.axis('off')
moupos = plt.ginput(4)
print("clicked", moupos)
print(moupos[0][1])
print([mp[1] for mp in moupos])
tp = np.array([[mp[1] for mp in moupos],
               [mp[0] for mp in moupos],
               [1, 1, 1, 1]])
im3 = warp.image_in_image(im1, im2, tp)
plt.figure()
plt.gray()
plt.imshow(im3)
plt.axis('equal')
plt.axis('off')
plt.show()


