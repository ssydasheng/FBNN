import numpy as np


__all__ = [
    'x3_toy',
    'x3_gap_toy',
    'sin_toy',
]


class toy_dataset(object):
    def __init__(self, name=''):
        self.name = name

    def train_samples(self):
        raise NotImplementedError

    def test_samples(self):
        raise NotImplementedError


class x3_toy(toy_dataset):
    def __init__(self, name='x3'):
        self.x_min = -6
        self.x_max = 6
        self.y_min = -100
        self.y_max = 100
        self.confidence_coeff = 3.
        self.f = lambda x: np.power(x, 3)
        self.y_std = 3.
        super(x3_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(1)

        X_train = np.random.uniform(-4, 4, (n_data, 1))
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(-6, 6, num=1000, dtype=np.float32)
        outputs = np.power(inputs, 3)
        return inputs, outputs


class x3_gap_toy(toy_dataset):
    def __init__(self, name='x3_gap'):
        self.x_min = -6
        self.x_max = 6
        self.y_min = -100
        self.y_max = 100
        self.confidence_coeff = 3.
        self.y_std = 3.
        self.f = lambda x: np.power(x, 3)
        super(x3_gap_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(1)

        X_train_1 = np.random.uniform(-4, -1, (n_data // 2, 1))
        X_train_2 = np.random.uniform(1, 4, (n_data // 2, 1))
        X_train = np.concatenate([X_train_1, X_train_2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(-6, 6, num=1000, dtype=np.float32)
        outputs = np.power(inputs, 3)
        return inputs, outputs


class sin_toy(toy_dataset):
    def __init__(self, name='sin'):
        self.x_min = -5
        self.x_max = 5
        self.y_min = -3.5
        self.y_max = 3.5
        self.confidence_coeff = 1.
        self.y_std = 2e-1

        def f(x):
            return 2 * np.sin(4*x)
        self.f = f
        super(sin_toy, self).__init__(name)

    def train_samples(self):
        np.random.seed(3)

        X_train1 = np.random.uniform(-2, -0.5, (10, 1))
        X_train2 = np.random.uniform(0.5, 2, (10, 1))
        X_train = np.concatenate([X_train1, X_train2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (20, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(self.x_min, self.x_max, 1000)
        return inputs, self.f(inputs)
