# coding=utf-8
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import homography
from matplotlib.tri.triangulation import *


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

    return (1 - alpha) * im2 + alpha * im1_t


def alpha_for_triangle(points, m, n):
    """ Creates alpha map of size (m,n)
    for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """
    alpha = np.zeros((m, n))
    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = np.linalg.solve(points, [i, j, 1])
            if min(x) > 0:  # all coefficients positive
                alpha[i, j] = 1
    return alpha


def triangulate_points(x, y):
    return Triangulation(x, y).get_masked_triangles()


def pw_affine(fromim, toim, fp, tp, tri):
    """ Warp triangular patches from an image.
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation. """
    im = toim.copy()
    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3
    # create image to warp to (needed if iterate colors)
    im_t = np.zeros(im.shape, 'uint8')
    for t in tri:
        # compute affine transformation
        H = homography.Haffine_from_points(tp[:, t], fp[:, t])
        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(
                    fromim[:, :, col], H[:2, :2], (H[0, 2], H[1, 2]), im.shape[:2])
        else:
            im_t = ndimage.affine_transform(
                fromim, H[:2, :2], (H[0, 2], H[1, 2]), im.shape[:2])
            # alpha for triangle
            alpha = alpha_for_triangle(tp[:, t], im.shape[0], im.shape[1])
            # add triangle to image
            im[alpha > 0] = im_t[alpha > 0]
    return im
