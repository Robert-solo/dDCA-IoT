import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def cal_pairwise_dist(x):
    '''
    计算pairwise距离，x是matrix
    :param x:
    :return:
    '''
    sum_x = np.sum(np.square(x),1)
    dist = np.add(np.add(-2*np.dot(x,x.T),sum_x).T,sum_x)
    # 返回任意两个点之间距离的平方
    return dist
def my_mds(data, n_dims):
    '''

    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :return: (n_samples, n_dim)
    '''
    n, d =data.shape
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    T1 = np.ones((n, n)) * np.sum(dist) / n ** 2
    T2 = np.sum(dist, axis=1, keepdims=True) / n
    T3 = np.sum(dist, axis=0, keepdims=True) / n

    B = -(T1 - T2 - T3 + dist) / 2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    print(picked_eig_vector.shape, picked_eig_val.shape)
