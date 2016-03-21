import numpy as np


def l2_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def pdist(matrix, axis=0, dist_type='l2'):
    """
    input: dist_type: l2
    """
    dist_type_dict = {'l2': l2_dist}
    dist_func = dist_type_dict[dist_type]
    matrix = matrix.T if axis == 1 else matrix
    n = matrix.shape[1]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            tmp_dist = dist_func(matrix[:, i], matrix[:, j])
            dist_matrix[i, j] = tmp_dist
            dist_matrix[j, i] = tmp_dist
    return dist_matrix


