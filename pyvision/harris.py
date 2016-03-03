# coding=utf-8
from scipy.ndimage import filters
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_harris_response(im, sigma=3):
    '''
    计算灰度图中每个像素点的 Harris corner detector 的响应函数
    :param im: image
    :param sigma: 高斯分布中的 sigma 方差值（钟型图的跨度）
    :return: 图像对应的 Harris 响应图（每个像素点的灰度卷积值）
    '''
    # derivatives
    imx = np.core.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    another = filters.gaussian_filter(im, (sigma, sigma), (0, 1))
    imy = np.core.zeros(im.shape)
    # imy = filters.gaussian_filter(im, (sigma, sigma), (1, 0))
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / (Wtr+1e-38)

def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    '''
    检测图像并返回 Harris corners
    :param harrisim:
    :param min_dist:
    :param threshold:
    :return:
    '''

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist),
            (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    '''
    显示找到的 Corners
    :param image: image #e.g. array(Image.open('example.jpg').convert('L'))
    :param filtered_coords:
    :return:
    '''

    plt.figure()
    plt.gray()
    plt.imshow(image)
    # something is strange here
    # strange solved because image is arrayed horizontally
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.plot([10,20,30],[3,2,1],'ro')
    # plt.axis('off')
    plt.show()

def get_descriptors(image, filtered_coords, wid=5):
    '''
    返回每个 filtered_coords（角点） 点的周围 width×2+1 im_1_filtered_coords范围内的灰度方格子图
    :param image:
    :param filtered_coords:
    :param wid:
    :return:
    '''

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc

def match(desc1, desc2, threshold=0.5):
    '''
    计算一对图像描述子的单位交叉相干性（nomalized cross-correlation）
    :param desc1:
    :param desc2:
    :param threshold:
    :return:
    '''

    n = len(desc1[0])
    print n
    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    print len(d), ' rows and ', len(d[0]), 'columns'

    duration_start = time.time()
    for i in range(len(desc1)):

        if i%10 == 0:
            duration_end = time.time()
            print 'processing', i, 'row', 'column of data'
            time_consumed = duration_end - duration_start
            print '剩余时间估计： ',time_consumed * (len(desc1) - i)
            duration_start = time.time()


        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = np.sum(d1*d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    print 'match finished and d: ', d
    ndx = np.argsort(-d)
    # 返回逐一匹配后最佳分的匹配位置 index
    matchscores = ndx[:, 0]
    return matchscores

def match_twosided(desc1, desc2, threshold=0.5):
    '''
    两边对称的匹配
    :param desc1:
    :param desc2:
    :param threshold:
    :return:
    '''

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    # 默认是 -1 只有值大于阈值才会被设置为对应 index
    ndx_12 = np.where(matches_12 >= 0)[0]

    # 删除非对称的匹配对
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    # 返回匹配对点
    return matches_12


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))
        plt.imshow(im3)
        cols1 = im1.shape[1]
        for i, m in enumerate(matchscores):
            if m > 0:
                plt.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
                plt.axis('off')





