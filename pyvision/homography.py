# coding=utf-8
import numpy as np


def normalize(points):
    '''
    homogeneous coordinates so that last row = 1
    :param points:
    :return:
    '''

    for row in points:
        row /= points[-1]

    return points


def make_homog(points):
    '''
    Convert a set of points to homogeneous coordinates
    :param points:
    :return:
    '''

    return np.vstack((points, np.ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    '''
    根据fp 和 tp 用DLT（direct linear transformation）算法进行同型映射变换
    这里的 H 是最小二乘解 homography
    :param fp:
    :param tp:
    :return:
    '''

    if fp.shape != tp.shape:
        # 如果映射点不一一对应，则报错
        raise RuntimeError('number of points do not match')

    # condition points
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = np.dot(C1, fp)
    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = np.dot(C2, tp)
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]
    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # normalize and return
    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    """
    Find H, affine transformation, such that
    tp is affine transf of fp.

    described in detail in Multiple_View_Geometry_in_Computer_Vision__2nd_Edition Richard Hartley

    """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # condition points
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1, fp)
    # --to points--
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2, tp)
    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))
    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))
    return H / H[2, 2]

class RansacModel(object):
    '''
    Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC
    '''

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        '''
        Fit homography to four selected correspondences
        :param data:
        :return:
        '''

        # transpose to fit H_from_points()
        data = data.T

        # from points
        fp = data[:3, :4]
        # to points
        tp = data[3:, :4]

        # fit homography and return
        return H_from_points(fp, tp)

    def get_error(self, data, H):
        '''
        Apply homography to all correspondences,
        return error for each transformed point
        :param data:
        :param H:
        :return:
        '''

        data = data.T

        # from points
        fp = data[:3]
        # target points
        tp = data[3:]

        #transfrom fp
        fp_transformed = np.dot(H, fp)

        #normalize homo coordinates
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2] # divide every component by w

        #return error per point
        return np.sqrt(np.sum((tp-fp_transformed)**2, axis=0))

def H_from_ransac(fp, tp, model, maxiter=1000, match_theshold=10):
    '''
    Robust estimation of homography H from point correspondence using RANSAC
    (ransac.py from http://www.scipy.org/Cookbook/RANSAC).
    :param fp: (3*n arrays) points in hom. coordinates.
    :param tp: (3*n arrays) points in hom. coordinates.
    :param model:
    :param maxiter:
    :param match_theshold:
    :return:
    '''
    import ransac
    # group corresponding points
    data = np.vstack((fp, tp))

    # compute H and return
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_theshold, 10, return_all=True)
    return H, ransac_data['inliers']

