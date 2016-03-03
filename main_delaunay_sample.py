# coding=utf-8
from pyvision import warp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.tri.triangulation import *

x,y = np.array(np.random.standard_normal((2,100)))
tri = Triangulation(x, y).get_masked_triangles()
plt.figure()
for t in tri:
    t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
    plt.plot(x[t_ext],y[t_ext],'r')
plt.plot(x,y,'*')
plt.axis('off')
plt.show()