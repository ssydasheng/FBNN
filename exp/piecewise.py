import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse

from utils.logging import get_logger
from core.fvi import KLEstimatorFVI
from utils.nets import get_posterior
from utils.utils import default_plotting_new as init_plotting
from utils.prior_gen import PiecewiseLinear, PiecewiseConstant

parser = argparse.ArgumentParser('Piecewise')
parser.add_argument('-d', '--dataset', type=str, default='p_lin')
parser.add_argument('-N', '--N', type=int, default=40)
parser.add_argument('-in', '--injected_noise', type=float, default=0.00)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)
parser.add_argument('-na', '--n_rand', type=int, default=100)
parser.add_argument('-nh', '--n_hidden', type=int, default=2)
parser.add_argument('-nu', '--n_units', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.003)
parser.add_argument('-e', '--epochs', type=int, default=30000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.9)
parser.add_argument('--train_samples', type=int, default=500)

parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)

parser.add_argument('--seed', type=int, default=8)
args = parser.parse_args()
logger = get_logger(args.dataset, 'results/%s/'%(args.dataset), __file__)
print = logger.info
tf.set_random_seed(args.seed)
np.random.seed(args.seed)
xmin, xmax = 0., 1.
ymin, ymax = 0., 1.
lambda_, y_std = 3., 0.02

############################## setup FBNN model ##############################
prior_generator = dict(p_lin=PiecewiseLinear, p_const=PiecewiseConstant)[args.dataset](
    lambda_, xmin, xmax, ymin, ymax).sample

def rand_generator(*arg):
    return tf.random_uniform(shape=[args.n_rand, 1], minval=xmin, maxval=xmax)

layer_sizes = [1] + [args.n_units] * args.n_hidden + [1]
model = KLEstimatorFVI(
    prior_generator, get_posterior('bnn_tanh')(layer_sizes, args.init_logstd), rand_generator=rand_generator,
    obs_var=y_std**2., input_dim=1, n_rand=args.n_rand, injected_noise=args.injected_noise,
    n_eigen_threshold=args.n_eigen_threshold)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

############################## generate and normalize data ##############################
train_x = tf.concat([
    tf.random_uniform(minval=0., maxval=0.2, shape=[args.N // 2, 1]),
    tf.random_uniform(minval=0.8, maxval=1.0, shape=[args.N // 2, 1])
], axis=0)
test_x = tf.reshape(tf.linspace(xmin, xmax, 2000), [-1, 1])
train_y = tf.squeeze(prior_generator(tf.concat([train_x, test_x], axis=0), 1))
train_y, test_y = train_y[:args.N], train_y[args.N:]
train_x_sample, train_y_sample, test_x_sample, test_y_sample = sess.run([train_x, train_y, test_x, test_y])
train_y_sample = train_y_sample + y_std * np.random.normal(size=train_y_sample.shape)

############################## training ##############################

for epoch in range(1, 1+args.epochs):
    feed_dict = {model.x: train_x_sample, model.y: train_y_sample, model.learning_rate_ph: args.learning_rate,
                 model.n_particles: args.train_samples}
    feed_dict.update(model.default_feed_dict())

    _, elbo_sur, kl_sur, logll = sess.run(
        [model.infer_latent, model.elbo, model.kl_surrogate, model.log_likelihood],
        feed_dict=feed_dict)

    if epoch % args.print_interval == 0:
        print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(
            epoch, args.epochs, elbo_sur, logll, kl_sur))

    if epoch % args.test_interval == 0:
        y_pred = sess.run(model.func_x_pred,
                          feed_dict={model.x_pred: np.reshape(test_x_sample, [-1, 1]),
                                     model.n_particles: args.test_samples})
        mean_y_pred, std_y_pred = np.mean(y_pred, 0), np.std(y_pred, 0)

        plt.clf()
        figure = plt.figure(figsize=(8, 5.5), facecolor='white')
        init_plotting()

        ## plt.plot(test_x_sample.squeeze(), mean_y_pred, 'steelblue', label='Mean function')
        plt.fill_between(test_x_sample.squeeze(),
                         mean_y_pred - 3. * std_y_pred,
                         mean_y_pred + 3. * std_y_pred, alpha=0.2, color='b')
        for id in range(4):
            plt.plot(test_x_sample.squeeze(), y_pred[id], 'g')
        plt.scatter(train_x_sample, train_y_sample, c='tomato', zorder=10, label='Observations')

        plt.grid(True)
        plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tight_layout()
        plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])
        plt.tight_layout()

        plt.savefig('results/{}/plot_epoch{}.pdf'.format(args.dataset, epoch))
