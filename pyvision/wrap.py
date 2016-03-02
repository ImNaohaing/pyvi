# coding=utf-8
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import homography

def image_in_image(im1, im2, tp):
    '''
    Put im1 in im2 with an affine transformation
    such that corners are as close to tp as possible.
    tp are homogeneous and counterclockwise from top left.
    :param im1: inner image
    :param im2: container image
    :param tp: homogeneous and counterclockwise from top left
    :return:
    '''

    # points to warp from
    m, n = im1.shape[:2]

    # from points array [[x], [y], [w]]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # to points array [[x], [y], [w]]
    H = homography.Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 +