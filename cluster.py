import os
import numpy as np
from PIL import Image, ImageDraw


class ClusterNode(object):
    def __init__(self, right=None, left=None, num=1, distance=0, nodeid=None):
        self.right = right
        self.left = left
        self.num = num
        self.distance = distance
        self.nodeid = nodeid


def shunxu(a, b):
    if a < b:
        return a, b
    else:
        return b, a


def cluster(data, method='single'):
    """
    input: method: 'single' or 'complete'
    output: cluster tree
    """
    methodFuncDict = {'single': np.min,
                      'complete': np.max,
                      }
    assert method in methodFuncDict
    methodFunc = methodFuncDict[method]
    distance = {}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance[(i, j)] = data[i, j]

    clust = [ClusterNode(nodeid=i) for i in range(len(data))]
    currentid = -1

    while len(clust) > 1:
        min_distance = 999
        min_pair = (-1, -1)

        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                d = distance[shunxu(clust[i].nodeid, clust[j].nodeid)]
                if d < min_distance:
                    min_distance = d
                    min_pair = (i, j)

        r, l = min_pair
        for i in range(len(clust)):
            if i != r and i != l:
                d1 = distance[shunxu(clust[r].nodeid, clust[i].nodeid)]
                d2 = distance[shunxu(clust[l].nodeid, clust[i].nodeid)]
                distance[shunxu(clust[i].nodeid, currentid)] = methodFunc((d1, d2))
        newnode = ClusterNode(right=clust[r], left=clust[l],
                              distance=min_distance, num=clust[r].num + clust[l].num, nodeid=currentid)
        currentid -= 1
        del clust[l]
        del clust[r]
        clust.append(newnode)
    return clust[0]


def getheight(clust):
    if clust.left is None and clust.right is None:
        return 1
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    if clust.left is None and clust.right is None:
        return 1
    return min(getdepth(clust.left), getdepth(clust.right), clust.distance)


def drawdendrogram(clust, labels, savePath=''):
    h = getheight(clust) * 20
    w = 1400
    # depth = getdepth(clust)
    scaling = 500

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # draw.line((0, h/2, 10, h/2),fill=(255, 0, 0))

    drawnode(draw, clust, 200 + clust.distance * scaling, (h / 2), scaling, labels)
    if savePath:
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        if not savePath.endswith('/'):
            savePath = savePath + '/'
        img.save(savePath + 'cluster.jpg', 'JPEG')
    else:
        img.show()


def drawnode(draw, clust, x, y, scaling, labels):
    if clust.nodeid < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        # line length
        lleft = clust.left.distance * scaling + 200
        lright = clust.right.distance * scaling + 200
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        draw.line((x, top + h1 / 2, lleft, top + h1 / 2), fill=(255, 0, 0))
        draw.line((x, bottom - h2 / 2, lright, bottom - h2 / 2), fill=(255, 0, 0))
        drawnode(draw, clust.left, lleft, top + h1 / 2, scaling, labels)
        drawnode(draw, clust.right, lright, bottom - h2 / 2, scaling, labels)
    else:
        draw.text((x - 10, y), str(labels[clust.nodeid]), (0, 0, 0))


if __name__ == '__main__':
    def filter_data(x):
        return [i for i in x if np.sum(np.asarray(i) == 0) != len(i)]


    from sklearn import datasets

    digits = datasets.load_digits()
    images = digits.images
    X = np.reshape(images, (len(images), -1))
    X = np.transpose(X)
    X = np.asarray(filter_data(X))

    # if you use corr matrix, please change it to distance matrix
    cx = 1 - np.abs(np.corrcoef(X))

    # cluster gives a cluster tree
    aa = cluster(cx)

    # save a cluster tree
    drawdendrogram(aa, range(1, 65), 'aaa')
