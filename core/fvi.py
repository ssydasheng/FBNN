import gpflowSlim as gfs
import tensorflow as tf

from core.grad_estimator import SpectralScoreEstimator, entropy_surrogate


class AbstractFVI(object):
    """
    Base Class for Functional Variational Inference.

    :param posterior: the posterior network to be optimized.
    :param rand_generator: Generates measurement points.
    :param obs_var: Float. Observation variance.
    :param input_dim. Int.
    :param n_rand. Int. Number of random measurement points.
    :param injected_noise: Float. Injected to function outputs for stability.
    """
    def __init__(self, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise):

        self.posterior = posterior
        self._rand_generator = rand_generator
        self.obs_var = obs_var

        self.input_dim = input_dim
        self.n_rand = n_rand
        self.injected_noise = injected_noise

        self.init_inputs()
        self.build_rand()
        self.build_function()
        self.build_log_likelihood()

    def init_inputs(self):
        self.x                = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
        self.x_pred           = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x_pred')
        self.y                = tf.placeholder(tf.float32, shape=[None], name='y')
        self.n_particles      = tf.placeholder(tf.int32,   shape=[], name='n_particles')
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.coeff_ll         = tf.placeholder(tf.float32, shape=[], name='coeff_ll')
        self.coeff_kl         = tf.placeholder(tf.float32, shape=[], name='coeff_kl')

    @property
    def batch_size(self):
        return tf.to_float(tf.shape(self.x)[0])

    def build_rand(self):
        self.rand = self._rand_generator(self)
        self.x_rand = tf.concat([self.x, self.rand], axis=0)

    def build_function(self):
        self.repeat_x_rand = tf.tile(tf.expand_dims(self.x_rand, 0), [self.n_particles, 1, 1])

        # [n_particles, batch_size + n_rand]
        self.func_x_rand = self.posterior(self.x_rand, self.n_particles)
        self.func_x = self.func_x_rand[:, :tf.shape(self.x)[0]]
        self.func_x_pred = self.posterior(self.x_pred, self.n_particles)

        self.noisy_func_x_rand = self.func_x_rand + self.injected_noise * tf.random_normal(shape=tf.shape(self.func_x_rand))

    def build_log_likelihood(self):
        y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1])
        y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
        self.log_likelihood = tf.reduce_mean(y_x_dist.log_prob(y_obs))
        self.y_x_pred = y_x_dist.sample()

    @property
    def params_posterior(self):
        return tf.trainable_variables('posterior')

    @property
    def params_prior(self):
        return tf.trainable_variables('prior')

    @property
    def params_likelihood(self):
        return tf.trainable_variables('likelihood')

    def build_kl(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * self.kl_surrogate / self.batch_size

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior) \
            if len(self.params_posterior) else tf.no_op()

        flag = (not len(self.params_prior)) and any(tf.gradients(self.elbo, self.params_prior))
        self.infer_prior = tf.no_op() if not flag else self.optimizer.minimize(-self.elbo)

        self.infer_likelihood = self.optimizer.minimize(-self.elbo, var_list=self.params_likelihood)\
            if len(self.params_likelihood) else tf.no_op()

        self.infer_joint = tf.group(self.infer_latent, self.infer_prior, self.infer_likelihood)

    def default_feed_dict(self):
        return {self.coeff_kl: 1., self.coeff_ll: 1.}


class EntropyEstimationFVI(AbstractFVI):
    """
    Function Variational Inference with estimating entropy and computing cross entropy analytically.
    """
    def __init__(self, prior_kernel, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise,
                 n_eigen_threshold=0.99, eta=0.):
        super(EntropyEstimationFVI, self).__init__(
            posterior, rand_generator, obs_var,
            input_dim, n_rand, injected_noise)
        self.n_eigen_threshold = n_eigen_threshold
        self.eta = eta

        self.prior_kernel = prior_kernel

        self.build_kl()
        self.build_optimizer()

    def build_kl(self):
        # estimate entropy surrogate
        estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)

        # compute analytic cross entropy
        kernel_matrix = self.prior_kernel.K(tf.cast(self.x_rand, tf.float64)) \
                        + self.injected_noise ** 2 * tf.eye(tf.shape(self.x_rand)[0], dtype=tf.float64)
        prior_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(
            tf.zeros([tf.shape(self.x_rand)[0]], dtype=tf.float64), kernel_matrix)
        cross_entropy = -tf.reduce_mean(prior_dist.log_prob(tf.to_double(self.noisy_func_x_rand)))

        self.kl_surrogate = -entropy_sur + tf.to_float(cross_entropy)

    def build_prior_gp(self, init_var=0.1, inducing_points=None):
        self.x_gp      = tf.placeholder(tf.float64, shape=[None, self.input_dim], name='x_gp')
        self.y_gp      = tf.placeholder(tf.float64, shape=[None], name='y_gp')
        self.x_pred_gp = tf.placeholder(tf.float64, shape=[None, self.input_dim], name='x_pred_gp')
        with tf.variable_scope('prior'):
            if inducing_points is None:
                self.gp = gfs.models.GPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel, obs_var=init_var)
            else:
                self.gp = gfs.models.SGPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel, Z=inducing_points)
            self.gp_loss = self.gp.objective

        self.gp_var = self.gp.likelihood.variance
        self.gp_logstd = tf.log(self.gp.likelihood.variance) * 0.5
        self.func_x_pred_gp = tf.squeeze(self.gp.predict_f_samples(self.x_pred_gp, self.n_particles), -1)

        self.optimizer_gp = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_gp = self.optimizer_gp.minimize(self.gp_loss, var_list=self.params_prior)\
            if len(self.params_prior) else tf.no_op()

        # only optimize kernel params without optimizing GP observation variance
        self.infer_gp_kern = self.optimizer_gp.minimize(
            self.gp_loss, var_list=[v for v in self.params_prior if 'likelihood' not in v.name])


class KLEstimatorFVI(AbstractFVI):
    """
    Function Variational Inference with estimating the whole KL divergence term.
    """
    def __init__(self, prior_generator, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise,
                 n_eigen_threshold=0.99, eta=0.):
        super(KLEstimatorFVI, self).__init__(
            posterior, rand_generator, obs_var,
            input_dim, n_rand, injected_noise)
        self.prior_gen = prior_generator
        self.n_eigen_threshold = n_eigen_threshold
        self.eta = eta

        self.build_kl()
        self.build_optimizer()

    def build_kl(self):
        # estimate entropy surrogate
        estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)

        # estimate cross entropy
        self.prior_func_x_rand = self.prior_gen(self.x_rand, self.n_particles)
        self.noisy_prior_func_x_rand = self.prior_func_x_rand + self.injected_noise * tf.random_normal(
            tf.shape(self.prior_func_x_rand))

        cross_entropy_gradients = estimator.compute_gradients(self.noisy_prior_func_x_rand,
                                                              self.noisy_func_x_rand)
        cross_entropy_sur = -tf.reduce_mean(tf.reduce_sum(
            tf.stop_gradient(cross_entropy_gradients) * self.noisy_func_x_rand, -1))

        self.kl_surrogate = -entropy_sur + cross_entropy_sur
