"""Functional Variational Bayesian neural networks (fBNNs)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import flags
import zhusuan as zs

from bandits.algorithms.multitask_gp import MultitaskGP
from bandits.core.bayesian_nn import BayesianNN
from core.grad_estimator import SpectralScoreEstimator, entropy_surrogate

FLAGS = flags.FLAGS


def log_gaussian(x, mu, sigma, reduce_sum=True):
  """Returns log Gaussian pdf."""
  res = (-0.5 * np.log(2 * np.pi) - tf.log(sigma) - tf.square(x - mu) /
         (2 * tf.square(sigma)))
  res = tf.reduce_mean(res, 0)
  if reduce_sum:
    return tf.reduce_sum(res)
  else:
    return res


class FunctionalBNNModel(BayesianNN):
  """Implements an functional Bayesian NN."""

  def __init__(self, hparams, name="FBNN"):

    self.name = name
    self.hparams = hparams

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_actions
    self.n_adv = self.hparams.n_adv
    self.layers = self.hparams.layer_sizes
    self.init_scale = self.hparams.init_scale
    self.n_sample = self.hparams.n_sample
    self.f_num_points = None
    if "f_num_points" in hparams:
      self.f_num_points = self.hparams.f_num_points

    self.cleared_times_trained = self.hparams.cleared_times_trained
    self.initial_training_steps = self.hparams.initial_training_steps
    self.training_schedule = np.linspace(self.initial_training_steps,
                                         self.hparams.training_epochs,
                                         self.cleared_times_trained)
    self.verbose = getattr(self.hparams, "verbose", True)

    self.times_trained = 0

    if self.hparams.use_sigma_exp_transform:
      self.sigma_transform = tf.exp
      self.inverse_sigma_transform = np.log
    else:
      self.sigma_transform = tf.nn.softplus
      self.inverse_sigma_transform = lambda y: y + np.log(1. - np.exp(-y))

    self.gp = MultitaskGP(hparams.hparams_gp)
    self.build_graph(self.gp.graph, self.gp.sess)

  def build_mu_variable(self, shape):
    """Returns a mean variable initialized as N(0, 0.05)."""
    return tf.Variable(tf.random_normal(shape, 0.0, 0.05))

  def build_sigma_variable(self, shape, init=-5.):
    """Returns a sigma variable initialized as N(init, 0.05)."""
    # Initialize sigma to be very small initially to encourage MAP opt first
    return tf.Variable(tf.random_normal(shape, init, 0.05))

  def build_layer(self, input_x, shape, layer_id, activation_fn=tf.nn.relu):
    """Builds a variational layer, and computes KL term.

    Args:
      input_x: Input to the variational layer.
      shape: [number_inputs, number_outputs] for the layer.
      layer_id: Number of layer in the architecture.
      activation_fn: Activation function to apply.

    Returns:
      output_h: Output of the variational layer.
    """
    w_mu = self.build_mu_variable(shape)
    w_sigma = self.sigma_transform(self.build_sigma_variable(shape))
    w_noise = tf.random_normal([self.n_sample] + shape)
    w = w_mu + w_sigma * w_noise

    b_mu = self.build_mu_variable([1, shape[1]])
    # b_sigma = self.sigma_transform(self.build_sigma_variable([1, shape[1]]))
    b = b_mu

    # Create outputs
    output_h = activation_fn(tf.matmul(input_x, w) + b)
    return output_h

  def build_model(self, activation_fn=tf.nn.relu):
    """Defines the actual NN model with fully connected layers.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.

    Args:
      activation_fn: the activation function used in the nn layers.
    """

    if self.verbose:
      print("Initializing model {}.".format(self.name))
    neg_kl_term, l_number = 0, 0

    # Build network.
    input_x = tf.tile(tf.expand_dims(self.x_adv, 0), [self.n_sample, 1, 1])
    n_in = self.n_in

    for l_number, n_nodes in enumerate(self.layers):
      if n_nodes > 0:
        h = self.build_layer(input_x, [n_in, n_nodes], l_number, self.hparams.activation)
        input_x = h
        n_in = n_nodes

    # Create last linear layer
    h = self.build_layer(input_x, [n_in, self.n_out],
                         l_number + 1, activation_fn=lambda x: x)

    self.func_x_adv = h # (h - tf.to_float(self.gp.input_mean)) / tf.to_float(self.gp.input_std)
    self.func_x = self.func_x_adv[:, :tf.shape(self.x)[0]]
    self.func_adv = self.func_x_adv[:, tf.shape(self.x)[0]:]
    self.y_pred = h[0, :tf.shape(self.x)[0]] * tf.to_float(self.gp.input_std) + tf.to_float(self.gp.input_mean)
    #TODO: mean

    self.injected_noise = 0.01
    noisy_func_x_adv = self.func_x_adv + self.injected_noise * tf.random_normal(tf.shape(h))
    tmp = tf.boolean_mask(tf.transpose(noisy_func_x_adv, [1, 2, 0]), self.weights_x_adv > 0)
    self.noisy_func_x_adv = tf.transpose(tmp)

    # alpha = tf.nn.softplus(self.gp.noise) + 1e-6
    # self.obs_sigma = tf.constant(self.hparams.noise_sigma)
    self.obs_sigma = tf.nn.softplus(tf.get_variable('pre_noise_sigma', initializer=-3.))
    y_normed = (self.y - tf.to_float(self.gp.input_mean)) / tf.to_float(self.gp.input_std)
    log_likelihood = log_gaussian(y_normed, self.func_x, self.obs_sigma, reduce_sum=False)

    # Compute functional kl divergence
    x_adv_64, w_x_adv_64 = tf.cast(self.x_adv, tf.float64), tf.cast(self.weights_x_adv, tf.float64)
    eye = tf.eye(tf.shape(self.x_adv)[0], dtype=tf.float64)
    prior_cov = self.gp.cov(x_adv_64, x_adv_64) * self.gp.task_cov(w_x_adv_64, w_x_adv_64) \
      + self.injected_noise ** 2 * eye
    prior_cov_root = tf.to_float(tf.linalg.cholesky(prior_cov))

    estimator = SpectralScoreEstimator(n_eigen_threshold=0.99)
    entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_adv)

    prior_dist = zs.distributions.MultivariateNormalCholesky(
        mean=tf.zeros([tf.shape(self.x_adv)[0]]),
        cov_tril=prior_cov_root)
    cross_entropy = tf.reduce_mean(prior_dist.log_prob(self.noisy_func_x_adv))
    kl = -entropy_sur - cross_entropy

    # Only take into account observed outcomes (bandits setting)
    batch_size = tf.to_float(tf.shape(self.x)[0])
    self.weighted_log_likelihood = tf.reduce_sum(
        log_likelihood * self.weights) / batch_size

    self.global_step = tf.train.get_or_create_global_step()
    kl_coeff = 1. # tf.minimum(tf.to_float(self.global_step % 20000) / 15000., 1.) #TODO
    elbo = self.weighted_log_likelihood - kl / batch_size * kl_coeff

    self.loss = -elbo
    vars = list(set(tf.trainable_variables()) - set(tf.trainable_variables(self.gp.name)))
    optimizer = tf.train.AdamOptimizer(self.hparams.initial_lr)
    gradients = optimizer.compute_gradients(self.loss, var_list=vars)
    clipped_grads = [(tf.clip_by_value(grad, -self.hparams.max_grad_norm, self.hparams.max_grad_norm), var)
        for grad, var in gradients]
    self.train_op = optimizer.apply_gradients(clipped_grads)

  def build_graph(self, graph, sess):
    """Defines graph, session, placeholders, and model.

    Placeholders are: n (size of the dataset), x and y (context and observed
    reward for each action), and weights (one-hot encoding of selected action
    for each context, i.e., only possibly non-zero element in each y).
    """

    self.graph = graph
    with self.graph.as_default():

      self.sess = sess
      self.x = tf.placeholder(shape=[None, self.n_in], dtype=tf.float32)
      self.y = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)
      self.weights = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)

      xmin, xmax = tf.reduce_min(self.x, 0), tf.reduce_max(self.x, 0) #TODO: adv region
      self.adv = tf.random_uniform(shape=[self.n_adv, self.n_in]) * (xmax-xmin) * 2 + xmin - (xmax-xmin)/2.
      rand = tf.to_int32(tf.random_uniform(shape=[self.n_adv], minval=-0.5, maxval=self.n_out-0.5))
      self.weights_adv = tf.one_hot(rand, self.n_out)

      self.x_adv = tf.concat([self.x, self.adv], axis=0)
      self.weights_x_adv = tf.concat([self.weights, self.weights_adv], axis=0)

      self.build_model()
      self.sess.run(tf.global_variables_initializer())

  def assign_lr(self):
    """Resets the learning rate in dynamic schedules for subsequent trainings.

    In bandits settings, we do expand our dataset over time. Then, we need to
    re-train the network with the new data. The algorithms that do not keep
    the step constant, can reset it at the start of each *training* process.
    """

    decay_steps = 1
    if self.hparams.activate_decay:
      current_gs = self.sess.run(self.global_step)
      with self.graph.as_default():
        self.lr = self.hparams.initial_lr

  def train(self, data, num_steps):
    """Trains the BNN for num_steps, using the data in 'data'.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.

    Returns:
      losses: Loss history during training.
    """

    if self.times_trained < self.cleared_times_trained:
      num_steps = int(self.training_schedule[self.times_trained])
    self.times_trained += 1

    losses = []
    with self.graph.as_default():
      if self.verbose:
        print("Training GP first ")
        self.gp.train(data, 100)

      if self.verbose:
        print("Training {} for {} steps...".format(self.name, num_steps))
      for step in range(num_steps):
        x, y, weights = data.get_batch_with_weights(self.hparams.batch_size, False)
        _, global_step, loss, ll, obss, yp = self.sess.run(
            [self.train_op, self.global_step, self.loss, self.weighted_log_likelihood, self.obs_sigma, self.y_pred],
            feed_dict={
                self.x: x,
                self.y: y,
                self.weights: weights,
            })

        losses.append(loss)
        if step % 100 == 0:
            print('Iter %d/%d -- ELBO %.4f -- Log Likelihood %.4f -- Obs Sigma %.4f' % (step, num_steps, -loss, ll, obss))
            print('Y Pred = {}'.format(np.choose(weights.argmax(1), yp.transpose()).reshape([1, -1])[0, :5]))
            print('Y True = {}'.format(np.choose(weights.argmax(1), y.transpose()).reshape([1, -1])[0, :5]))
    print('------End Training for one Epoch-----')
    return losses
