import tensorflow as tf
import zhusuan as zs


def get_posterior(name):
    if name == 'bnn' or name == 'bnn_relu':
        return bnn_outer(tf.nn.relu)
    if name == 'bnn_tanh':
        return bnn_outer(tf.nn.tanh)
    raise NameError('Not a supported name for posterior')

def bnn_outer(activation):
    def bnn_inner(layer_sizes, logstd_init=-5.):
        @zs.reuse('posterior')
        def bnn(x, n_particles):
            # x: [batch_size, input_dim]
            h = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])

            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mean = tf.get_variable('w_mean_'+str(i), shape=[n_in, n_out],
                                         initializer = tf.contrib.layers.xavier_initializer())
                w_logstd = tf.get_variable('w_logstd_'+str(i), shape=[n_in, n_out],
                                           initializer=tf.constant_initializer(logstd_init))
                w_std = tf.exp(w_logstd)
                ws = w_mean + w_std * tf.random_normal([n_particles, n_in, n_out])

                b_mean = tf.get_variable('b_mean_' + str(i), shape=[1, n_out],
                                         initializer=tf.zeros_initializer())
                b_logstd = tf.get_variable('b_logstd_' + str(i), shape=[1, n_out],
                                           initializer=tf.constant_initializer(logstd_init))
                b_std = tf.exp(b_logstd)
                bs = b_mean + b_std * tf.random_normal([n_particles, 1, n_out])

                h = tf.matmul(h, ws) + bs
                if i < len(layer_sizes) - 2:
                    h = activation(h)
            h = tf.squeeze(h, -1)
            # h: [n_particles, N]
            return h
        return bnn

    return bnn_inner