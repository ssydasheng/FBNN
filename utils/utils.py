import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def default_plotting_new():
   plt.rcParams['font.size'] = 15
   plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['axes.ymargin'] = 0
   plt.rcParams['axes.xmargin'] = 0

def default_plotting():
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def get_kemans_init(x, k_centers):
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]

    kmeans = MiniBatchKMeans(n_clusters=k_centers, batch_size=k_centers*10).fit(x)
    return kmeans.cluster_centers_


def median_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.median(dis_a)


def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)

