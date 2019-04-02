import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gpflowSlim as gfs
float_type = gfs.settings.tf_float
import argparse

from utils.logging import get_logger
from core.fvi import EntropyEstimationFVI
from utils.nets import get_posterior
from utils.utils import default_plotting_new as init_plotting
from data import x3_gap_toy, sin_toy


parser = argparse.ArgumentParser('Toy')
parser.add_argument('-d', '--dataset', type=str, default='x3')
parser.add_argument('-in', '--injected_noise', type=float, default=0.01)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)
parser.add_argument('-na', '--n_rand', type=int, default=20)
parser.add_argument('-nh', '--n_hidden', type=int, default=2)
parser.add_argument('-nu', '--n_units', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=10000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--train_samples', type=int, default=100)

parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)

args = parser.parse_args()
logger = get_logger(args.dataset, 'results/%s/'%args.dataset, __file__)
print = logger.info


############################## load and normalize data ##############################
dataset = dict(x3=x3_gap_toy, sin=sin_toy)[args.dataset]()
original_x_train, original_y_train = dataset.train_samples()
mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)
mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
train_x = (original_x_train - mean_x) / std_x
train_y = (original_y_train - mean_y) / std_y
original_x_test, original_y_test = dataset.test_samples()
test_x = (original_x_test - mean_x) / std_x
test_y = (original_y_test - mean_y) / std_y

y_logstd = np.log(dataset.y_std / std_y)

lower_ap = (dataset.x_min - mean_x) / std_x
upper_ap = (dataset.x_max - mean_x) / std_x


############################## setup FBNN model ##############################
with tf.variable_scope('prior'):
    if args.dataset == 'x3':
        prior_kernel = gfs.kernels.Linear(input_dim=1, name='lin') + gfs.kernels.RBF(input_dim=1, name='rbf')
    elif args.dataset == 'sin':
        prior_kernel = gfs.kernels.Periodic(input_dim=1, name='per') + gfs.kernels.RBF(input_dim=1, name='rbf')

def rand_generator(*arg):
    return tf.random_uniform(shape=[args.n_rand, 1], minval=lower_ap, maxval=upper_ap)

layer_sizes = [1] + [args.n_units] * args.n_hidden + [1]
model = EntropyEstimationFVI(
    prior_kernel, get_posterior('bnn_relu')(layer_sizes, args.init_logstd), rand_generator=rand_generator,
    obs_var=tf.exp(2.*y_logstd), input_dim=1, n_rand=args.n_rand, injected_noise=args.injected_noise)
model.build_prior_gp(np.exp(2 * y_logstd))


############################## training #######################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

gp_epochs = 10000
for epoch in range(gp_epochs):
    feed_dict = {model.x_gp:train_x, model.y_gp:train_y, model.learning_rate_ph: 0.003}
    _, loss = sess.run([model.infer_gp_kern, model.gp_loss], feed_dict=feed_dict)
    if epoch % args.print_interval == 0:
        print('>>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f}'.format(epoch, gp_epochs, loss))

for epoch in range(args.epochs):
    feed_dict = {model.x: train_x, model.y: train_y, model.learning_rate_ph: args.learning_rate,
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
                          feed_dict={model.x_pred: np.reshape(test_x, [-1, 1]),
                                     model.n_particles: args.test_samples})
        y_pred = y_pred * std_y + mean_y
        mean_y_pred, std_y_pred = np.mean(y_pred, 0), np.std(y_pred, 0)

        plt.clf()
        figure = plt.figure(figsize=(8, 5.5), facecolor='white')
        init_plotting()

        plt.plot(original_x_test.squeeze(), original_y_test, 'g', label="True function")
        plt.plot(original_x_test.squeeze(), mean_y_pred, 'steelblue', label='Mean function')
        for i in range(5):
            plt.fill_between(original_x_test.squeeze(), mean_y_pred - i * 0.75 * std_y_pred,
                             mean_y_pred - (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
            plt.fill_between(original_x_test.squeeze(), mean_y_pred + i * 0.75 * std_y_pred,
                             mean_y_pred + (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
        plt.scatter(original_x_train, original_y_train, c='tomato', zorder=10, label='Observations')
        plt.grid(True)
        plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tight_layout()
        plt.ylim([dataset.y_min, dataset.y_max])
        plt.tight_layout()

        plt.savefig('results/{}/plot_epoch{}.pdf'.format(args.dataset, epoch))