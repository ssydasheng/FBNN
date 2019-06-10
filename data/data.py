import os
import sys

from .hparams import HParams
from .register import register

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import numpy as np
import matplotlib.image as mpimg
import copy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data_path = os.path.join(root_path, 'data', 'uci')
DATASETS = dict(
    boston='housing.data',
    concrete='concrete.data',
    energy='energy.data',
    kin8nm='kin8nm.data',
    naval='naval.data',
    power_plant='power_plant.data',
    wine='wine.data',
    yacht='yacht_hydrodynamics.data',
    protein='protein.data',
    gpu='gpu.data',
    year='year_prediction.data',
    uk='uk.data',
    video_mem='video_mem.data',
    video_time='video_time.data',
)


@register('uci_woval')
def uci_woval(dataset_name, seed=1):
    data = np.loadtxt(os.path.join(data_path, DATASETS[dataset_name]))

    x, y = data[:, :-1], data[:, -1]
    if dataset_name == 'year':
        x_t, x_v, y_t, y_v = x[:463715], x[463715:], y[:463715], y[463715:]
    else:
        x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=seed)

    x_t, x_v, _, _ = standardize(x_t, x_v)
    y_t, y_v, _, train_std = standardize(y_t, y_v)

    hparams = HParams(
        x_train=x_t,
        x_test=x_v,
        y_train=y_t,
        y_test=y_v,
        std_y_train=train_std
    )
    return hparams


def standardize(data_train, *args):
    """
    Standardize a dataset to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output
