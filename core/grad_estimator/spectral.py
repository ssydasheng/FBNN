#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .base import ScoreEstimator


class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super(SpectralScoreEstimator, self).__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        # eigen_vectors: [..., M, n_eigen]
        # eigen_values: [..., n_eigen]
        # return: [..., N, n_eigen], by default n_eigen=M.
        M = tf.shape(samples)[-2]
        # Kxq: [..., N, M]
        # grad_Kx: [..., N, M, x_dim]
        # grad_Kq: [..., N, M, x_dim]
        Kxq = self.gram(x, samples, kernel_width)
        # Kxq = tf.Print(Kxq, [tf.shape(Kxq)], message="Kxq:")
        # ret: [..., N, n_eigen]
        ret = tf.sqrt(tf.to_float(M)) * tf.matmul(Kxq, eigen_vectors)
        ret *= 1. / tf.expand_dims(eigen_values, axis=-2)
        return ret

    def pick_n_eigen(self, eigen_values, eigen_vectors, n_eigen):
        # eigen_values: [..., M]
        # eigen_vectors: [..., M, M]
        M = tf.shape(eigen_values)[-1]
        # eigen_values: [..., n_eigen]
        # top_k_indices: [..., n_eigen]
        eigen_values, top_k_indices = tf.nn.top_k(eigen_values,
                                                  k=n_eigen)
        # eigen_values = tf.Print(eigen_values, [eigen_values],
        #                         "eigen_values:", summarize=10)
        # eigen_vectors_flat: [... * M, M]
        eigen_vectors_flat = tf.reshape(
            tf.matrix_transpose(eigen_vectors), [-1, M])
        # eigen_vectors_flat = tf.Print(eigen_vectors_flat,
        #                               [tf.shape(eigen_vectors_flat)],
        #                               message="eigen_vectors_flat:")
        # indices_2d: [..., n_eigen]
        indices_2d = tf.reshape(top_k_indices, [-1, n_eigen])
        # indices_2d = tf.Print(indices_2d, [tf.shape(indices_2d)],
        #                       message="indices_2d:")
        indices_2d += tf.range(tf.shape(indices_2d)[0])[..., None] * M
        # indices_2d = tf.Print(indices_2d, [tf.shape(indices_2d)],
        #                       message="indices_2d:")
        # indices_flat: [... * n_eigen]
        indices_flat = tf.reshape(indices_2d, [-1])
        # indices_flat = tf.Print(indices_flat, [tf.shape(indices_flat)],
        #                         message="indices_flat")
        # eigen_vectors_flat: [... * n_eigen, M]
        eigen_vectors_flat = tf.gather(eigen_vectors_flat, indices_flat)
        eigen_vectors = tf.matrix_transpose(
            tf.reshape(eigen_vectors_flat,
                       tf.concat([tf.shape(top_k_indices), [M]], 0)))
        # eigen_vectors = tf.Print(eigen_vectors, [tf.shape(eigen_vectors)],
        #                          message="eigen_vectors:", summarize=20)
        # eigen_vectors: [..., M, n_eigen]
        return eigen_values, eigen_vectors

    def compute_gradients(self, samples, x=None):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            # TODO: Simplify computation
            x = samples
        else:
            # _samples: [..., N + M, x_dim]
            _samples = tf.concat([samples, x], axis=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = tf.shape(samples)[-2]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * tf.eye(M)
        # eigen_vectors: [..., M, M]
        # eigen_values: [..., M]
        with tf.device("/cpu:0"):
            eigen_values, eigen_vectors = tf.self_adjoint_eig(Kq)
        # eigen_vectors = tf.matrix_inverse(Kq)
        # eigen_values = tf.reduce_sum(Kq, -1)
        # eigen_values = tf.Print(eigen_values, [eigen_values],
        #                         message="eigen_values:", summarize=20)
        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = tf.reduce_mean(
                tf.reshape(eigen_values, [-1, M]), axis=0)
            eigen_arr = tf.reverse(eigen_arr, axis=[-1])
            eigen_arr /= tf.reduce_sum(eigen_arr)
            eigen_cum = tf.cumsum(eigen_arr, axis=-1)
            self._n_eigen = tf.reduce_sum(tf.to_int32(tf.less(eigen_cum, self._n_eigen_threshold)))
            # self._n_eigen = tf.Print(self._n_eigen, [self._n_eigen],
            #                          message="n_eigen:")
        if self._n_eigen is not None:
            # eigen_values: [..., n_eigen]
            # eigen_vectors: [..., M, n_eigen]
            # eigen_values, eigen_vectors = self.pick_n_eigen(
            #     eigen_values, eigen_vectors, self._n_eigen)
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(
            samples, x, eigen_vectors, eigen_values, kernel_width)
        # grad_K1_avg = [..., M, x_dim]
        grad_K1_avg = tf.reduce_mean(grad_K1, axis=-3)
        # beta: [..., n_eigen, x_dim]
        beta = -tf.sqrt(tf.to_float(M)) * tf.matmul(
            eigen_vectors, grad_K1_avg, transpose_a=True) / tf.expand_dims(
            eigen_values, -1)
        # grads: [..., N, x_dim]
        grads = tf.matmul(eigen_ext, beta)
        return grads