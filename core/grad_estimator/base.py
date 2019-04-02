#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ScoreEstimator(object):
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        # square1 = tf.expand_dims(tf.reduce_sum(x1**2, -1), -1)
        # square2 = tf.expand_dims(tf.reduce_sum(x2**2, -1), -2)
        # len_x2 = len(x2.get_shape())
        # cross_term = tf.matmul(
        #     x1, tf.transpose(x2, range(len_x2-2)+[len_x2-1, len_x2-2]))
        # diff_square = square1 + square2 - 2 * cross_term
        # return tf.exp(-diff_square / (2 * kernel_width ** 2))
        return tf.exp(-tf.reduce_sum(tf.square((x1 - x2) / kernel_width), axis=-1) / 2)

    def gram(self, x1, x2, kernel_width):
        # x1: [..., n1, x_dim]
        # x2: [..., n2, x_dim]
        # kernel_width: [..., 1, 1, x_dim]
        # return: [..., n1, n2]
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        # x1: [..., n1, x_dim]
        # x2: [..., n2, x_dim]
        # kernel_width: [..., 1, 1, x_dim]
        # return gram, grad_x1, grad_x2:
        #   [..., n1, n2], [..., n1, n2, x_dim], [..., n1, n2, x_dim]
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        # G: [..., n1, n2]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        # diff: [..., n1, n2, n_x]
        diff = (x_row - x_col) / (kernel_width ** 2)
        # G_expand: [..., n1, n2, 1]
        G_expand = tf.expand_dims(G, axis=-1)
        # grad_x1: [..., n1, n2, n_x]
        grad_x2 = G_expand * diff
        # grad_x2: [..., n1, n2, n_x]
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        # x_samples: [..., n_samples, x_dim]
        # x_basis: [..., n_basis, x_dim]
        # return: [..., 1, 1, x_dim]
        x_dim = tf.shape(x_samples)[-1]
        n_samples = tf.shape(x_samples)[-2]
        n_basis = tf.shape(x_basis)[-2]
        x_samples_expand = tf.expand_dims(x_samples, -2)
        x_basis_expand = tf.expand_dims(x_basis, -3)
        pairwise_dist = tf.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.get_shape())
        reshape_dims = list(range(length-3)) + [length-1, length-3, length-2]
        pairwise_dist = tf.transpose(pairwise_dist, reshape_dims)

        k = n_samples * n_basis // 2
        top_k_values = tf.nn.top_k(
            tf.reshape(pairwise_dist, [-1, x_dim, n_samples * n_basis]),
            k=k).values

        kernel_width = tf.reshape(top_k_values[:, :, -1],
                                  tf.concat([tf.shape(x_samples)[:-2], [1, 1, x_dim]], axis=0))
        kernel_width = kernel_width * (tf.to_float(x_dim) ** 0.5)
        # kernel_width = tf.Print(kernel_width, [kernel_width],
        #                         message="kernel_width: ")
        kernel_width = kernel_width + tf.to_float(kernel_width < 1e-6) * 1.
        return tf.stop_gradient(kernel_width)

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()