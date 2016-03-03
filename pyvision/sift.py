# coding=utf-8
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

''' -- SIFT scale invariant feature transform --
http://en.wikipedia.org/wiki/Scale-invariant_feature_transform
'''
def process_image(imagename, resultname, params='--edge-thresh 10 --peak-thresh 5'):
    '''
    Process an image and save the results in a file
    :param imagename:
    :param resultname:
    :param params:
    :return:
    '''

    tmpfile_name = './tmp/tmp.pgm'
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save(tmpfile_name)
        imagename = tmpfile_name

    cmmd = str('%s %s --output=%s %s' %(os.path.abspath('./bin/sift'), imagename, resultname, params))
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname
    os.remove(os.path.abspath(tmpfile_name))
    print 'temp file removed'

def read_features_from_file(filename):
    '''
    读取 vlfeat 处理后的数据文件
    :param filename:
    :return:
    '''

    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]

def write_features_to_file(filename, locs, desc):
    '''

    :param filename:
    :param locs:
    :param desc:
    :return:
    '''
    np.savetxt(filename, np.hstack((locs, desc)))
    # all the same
    # np.savetxt(filename+'_v1', np.concatenate((locs, desc), axis=1))

def plot_features(im, locs, circle=False):
    '''
    特征显示在图片
    :param im: im as array
    :param locs: features locations
    :param circle: use circle or not
    :return:
    '''

    def draw_circle(c, r):
        '''
        :param c: center
        :param r: radius
        :return:
        '''

        t = np.arange(0, 1.01, .01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], 'ob')
    plt.axis('off')

def match(desc1, desc2):
    '''
    匹配两图之间的描述点
    :param desc1:
    :param desc2:
    :return:
    '''

    # 对每个 interest point 的 descriptor 进行 normalize
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t) # 比较两个向量之间的角度
        dotprods = 0.9999*dotprods

        # 对角度进行排序，第一个为向量角度最小的 interest points match，也就是最相似的点对
        indx = np.argsort(np.arccos(dotprods))

        # 并且要求这对 interest points match 向量角度 小于 第二相似点对的向量角度的 dist_ratio（0.6）倍
        # 也就是说相似度要达到第二候选的1.2倍 才能认为是绝对匹配的，形象的来说就是：两个人长得非常像而且任何一个人和其他人长得一点都不像
        if np.arccos(dotprods)[indx[0]] < dist_ratio*np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores

def match_twosided(desc1, desc2):
    '''
    交叉比较剔除计算上的误差导致的同一点对算出来的不同结果
    :param desc1:
    :param desc2:
    :return:
    '''

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # 剔除不对称的 match
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

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

def plot_matches3(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1, im2)
    cols1 = im1.shape[1]
    rowsmax = np.maximum(im1.shape[0], im2.shape[0])
    plt.plot([p[0] for p in locs1], [p[1] for p in locs1], 'o')
    plt.plot([p[0]+cols1 for p in locs2], [p[1] for p in locs2], 'o')
    if show_below:
        im3 = np.vstack((im3, im3, im3))
        plt.imshow(im3)
        for i, m in enumerate(matchscores):
            if m > 0:
                plt.plot([locs1[i][0], locs2[m][0] + cols1], [locs1[i][1]+rowsmax, locs2[m][1]+rowsmax], 'c')
                plt.axis('off')


def plot_matches_show3(ims, locs, matchscores):
    '''
    Show figures with lines joining the matches points
    :param ims: list of arrays(numpy array shapes) of images
    :param locs: list of locations(numpy array shapes) of interesting points
    :param matchscores: list of matching points(numpy array shapes) index
    :return:
    '''
    if ims.count < 2:
        raise RuntimeError('number of comparing images should bigger than one')
    imall = ims[0]
    for i in range(1, len(ims)):
        imall = appendimages(imall, ims[i])

    shapes = np.array([im.shape for im in ims])
    rowsmax = np.max([shape[0] for shape in shapes])
    imall = np.vstack((imall, imall, imall))
    plt.imshow(imall)
    # plot points
    for i, loc in enumerate(locs):
        plt.plot([p[0]+sum(shapes[:i, 1]) for p in loc], [p[1] for p in loc], 'o')
    for j, ms in enumerate(matchscores):
        for i, m in enumerate(ms[:, 0]):
                if m > 0:
                    plt.plot([locs[j][i][0] + sum(shapes[:j, 1]), locs[j+1][m][0] + sum(shapes[:j+1, 1])], [locs[j][i][1]+rowsmax, locs[j+1][m][1]+rowsmax], 'c')
                    plt.axis('off')





