# coding=utf-8
from scipy import linalg
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import homography
from matplotlib.tri.triangulation import *


def compute_rigid_transform(refpoints, points):
    """ Computes rotation, scale and translation for
    aligning points to refpoints. """
    A = np.array([[points[0], -points[1], 1, 0],
                  [points[1], points[0], 0, 1],
                  [points[2], -points[3], 1, 0],
                  [points[3], points[2], 0, 1],
                  [points[4], -points[5], 1, 0],
                  [points[5], points[4], 0, 1]])

    y = np.array([refpoints[0],
                  refpoints[1],
                  refpoints[2],
                  refpoints[3],
                  refpoints[4],
                  refpoints[5]])
    # least sq solution to mimimize ||Ax - y||
    a, b, tx, ty = linalg.lstsq(A, y)[0]
    R = np.array([[a, -b], [b, a]])  # rotation matrix incl scale

    return R, tx, ty
