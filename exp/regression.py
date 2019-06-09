import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np

import gpflowSlim as gfs
float_type = gfs.settings.tf_float
import argparse

from utils.logging import get_logger
from core.fvi import EntropyEstimationFVI
from utils.nets import get_posterior
from data import uci_woval
from utils.utils import median_distance_local


parser = argparse.ArgumentParser('Regression')
parser.add_argument('-d', '--dataset', type=str, default='boston')
parser.add_argument('-in', '--injected_noise', type=float, default=0.01)
parser.add_argument('-r', '--rand', type=str, default='uniform')
parser.add_argument('-na', '--n_rand', type=int, default=5)
parser.add_argument('-nh', '--n_hidden', type=int, default=1)
parser.add_argument('-nu', '--n_units', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=2000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--train_samples', type=int, default=100)

parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
logger = get_logger(args.dataset, 'results/regression/%s/'%args.dataset, __file__)
print = logger.info

def run(seed):
    tf.reset_default_graph()

    ############################## load and normalize data ##############################
    dataset = uci_woval(args.dataset, seed=seed)
    train_x, test_x, train_y, test_y = dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test
    std_y_train = dataset.std_y_train[0]
    N, input_dim = train_x.shape

    lower_ap = np.minimum(np.min(train_x), np.min(test_x))
    upper_ap = np.maximum(np.max(train_x), np.max(test_x))
    mean_x_train, std_x_train = np.mean(train_x, 0), np.std(train_x, 0)

    ############################## setup FBNN model ##############################
    with tf.variable_scope('prior'):
        ls = median_distance_local(train_x).astype('float32')
        ls[abs(ls) < 1e-6] = 1.
        prior_kernel = gfs.kernels.RBF(input_dim=input_dim, name='rbf', lengthscales=ls, ARD=True)

    with tf.variable_scope('likelihood'):
        obs_logstd = tf.get_variable('obs_logstd', shape=[], initializer=tf.constant_initializer(np.log(0.5)))
        obs_var = tf.exp(2.*obs_logstd)

    def rand_generator(*arg):
        if args.rand == 'uniform':
            return tf.random_uniform(shape=[args.n_rand, input_dim], minval=lower_ap, maxval=upper_ap)
        elif args.rand == 'normal':
            return mean_x_train + std_x_train * tf.random_normal(shape=[args.n_rand, input_dim])
        else:
            raise NotImplementedError

    layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
    model = EntropyEstimationFVI(
        prior_kernel, get_posterior('bnn')(layer_sizes, logstd_init=-2.), rand_generator=rand_generator,
        obs_var=obs_var, input_dim=input_dim, n_rand=args.n_rand, injected_noise=args.injected_noise)
    model.build_prior_gp(init_var=0.1)
    update_op = tf.group(model.infer_latent, model.infer_likelihood)
    with tf.control_dependencies([update_op]):
        train_op = tf.assign(obs_logstd, tf.maximum(tf.maximum(
            tf.to_float(model.gp_logstd), obs_logstd), tf.log(0.05)))

    ############################## training #######################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from scipy import stats

    gp_epochs = 5000
    for epoch in range(gp_epochs):
        feed_dict = {model.x_gp:train_x, model.y_gp:train_y, model.learning_rate_ph: 0.003}
        _, loss, gp_var = sess.run([model.infer_gp, model.gp_loss, model.gp_var], feed_dict=feed_dict)
        if epoch % args.print_interval == 0:
            print('>>> Seed {:5d} >>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f} | Var={:.5f}'.format(
                seed, epoch, gp_epochs, loss, gp_var))

            func_, var_ = sess.run([model.func_x_pred_gp, model.gp_var], feed_dict={model.x_gp:train_x, model.y_gp:train_y, model.x_pred_gp: test_x, model.n_particles: args.test_samples})
            mean_, var_ = np.mean(func_, 0), np.std(func_, 0)**2.+var_
            rmse = np.mean((mean_ - test_y) ** 2) ** .5 * std_y_train
            log_likelihood = np.mean(np.log(stats.norm.pdf(
                test_y,
                loc=mean_,
                scale=var_ ** 0.5))) - np.log(std_y_train)
            print('rmse = {}, logll = {}'.format(rmse, log_likelihood))

    epoch_iters = max(N // args.batch_size, 1)
    for epoch in range(1, args.epochs+1):
        indices = np.random.permutation(N)
        train_x, train_y = train_x[indices], train_y[indices]
        for iter in range(epoch_iters):
            x_batch = train_x[iter * args.batch_size: (iter + 1) * args.batch_size]
            y_batch = train_y[iter * args.batch_size: (iter + 1) * args.batch_size]

            feed_dict = {model.x: x_batch, model.y: y_batch, model.learning_rate_ph: args.learning_rate,
                         model.n_particles: args.train_samples}
            feed_dict.update(model.default_feed_dict())

            sess.run(train_op, feed_dict=feed_dict)

        if epoch % args.test_interval == 0 or epoch == args.epochs:
            feed_dict = {model.x: test_x, model.y: test_y, model.n_particles: args.test_samples}
            rmse, lld, ov = sess.run([model.eval_rmse, model.eval_lld, obs_var], feed_dict=feed_dict)
            rmse = rmse * std_y_train
            lld = lld - np.log(std_y_train)

            print('>>> Seed {:5d} >>> Epoch {:5d}/{:5d} | rmse={:.5f} | lld={:.5f} | obs_var={:.5f}'.format(
                seed, epoch, args.epochs, rmse, lld, ov))
            if epoch == args.epochs:
                return rmse, lld

if __name__ == '__main__':
    n_run = 1# 10 #TODO
    rmse_results, lld_results = [], []
    for seed in range(args.seed, n_run+args.seed):
        rmse, ll = run(seed)
        rmse_results.append(rmse)
        lld_results.append(ll)

    print("BNN test rmse = {}/{}".format(np.mean(rmse_results), np.std(rmse_results) / n_run ** 0.5))
    print("BNN test log likelihood = {}/{}".format(np.mean(lld_results), np.std(lld_results) / n_run ** 0.5))
    print('NOTE: Test result above output mean and std. errors')
