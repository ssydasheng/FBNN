import tensorflow as tf


class PiecewiseConstant(object):
    def __init__(self, lambda_, xmin, xmax, ymin, ymax):
        self.lambda_ = lambda_
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def sample(self, x, n_particles):
        # for example
        # loc = [0.2, 0.5]
        # val = [1.4, 1.6, 2]
        # [0, 0.2] --> 1.4; [0.2, 0.5] --> 1.6; [0.5, 1] --> 2

        # x: [N, 1]
        dist = tf.contrib.distributions.Poisson(self.lambda_)
        n_intervals = tf.to_int32(dist.sample(n_particles))

        res = tf.zeros([tf.shape(x)[0], 0])
        id = tf.constant(0)
        condition = lambda id, res: id < n_particles

        def body(id, res):
            loc = tf.random_uniform(minval=self.xmin, maxval=self.xmax, shape=[n_intervals[id]])
            val = tf.random_uniform(minval=self.ymin, maxval=self.ymax, shape=[n_intervals[id] + 1, 1])
            x_parts = tf.reduce_sum(tf.to_int32(x > loc), axis=-1)
            # [N, 1]
            y = tf.gather(val, x_parts)
            res = tf.concat([res, y], axis=1)
            return [id + 1, res]

        _, res = tf.while_loop(condition, body, loop_vars=[id, res],
                               shape_invariants=[id.get_shape(), tf.TensorShape([None, None])])

        res = tf.transpose(res)
        res.set_shape((None, None))
        return res


class PiecewiseLinear(object):
    def __init__(self, lambda_, xmin, xmax, ymin, ymax):
        self.lambda_ = lambda_
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def sample(self, x, n_particles):
        # for example
        # loc = [0.2, 0.5]
        # val = [1.4, 1.6, 2]
        # [0, 0.2] --> 1.4; [0.2, 0.5] --> 1.6; [0.5, 1] --> 2

        # x: [N, 1]
        dist = tf.contrib.distributions.Poisson(self.lambda_)
        n_intervals = tf.to_int32(dist.sample(n_particles))

        res = tf.zeros([tf.shape(x)[0], 0])
        id = tf.constant(0)
        condition = lambda id, res: id < n_particles

        def body(id, res):
            loc = tf.random_uniform(minval=self.xmin, maxval=self.xmax, shape=[n_intervals[id]])
            loc, _ = tf.nn.top_k(-loc, n_intervals[id])
            loc = - loc
            loc = tf.concat([[self.xmin], loc, [self.xmax]], axis=0)
            val = tf.random_uniform(minval=self.ymin, maxval=self.ymax, shape=[n_intervals[id] + 1, 1])

            # [N]
            id_left = tf.reduce_sum(tf.to_int32(x > loc), axis=-1) - 1
            id_right = id_left + 1
            # [N]
            x_left, x_right = tf.gather(loc, id_left), tf.gather(loc, id_right)
            # [N, 1]
            y_left, y_right = tf.gather(val, id_left), tf.gather(val, id_right)
            ratio = tf.expand_dims((tf.squeeze(x, 1) - x_left) / (x_right - x_left), 1)
            y = ratio * (y_right - y_left) + y_left

            res = tf.concat([res, y], axis=1)
            return [id + 1, res]

        _, res = tf.while_loop(condition, body, loop_vars=[id, res],
                               shape_invariants=[id.get_shape(), tf.TensorShape([None, None])])

        res = tf.transpose(res)
        res.set_shape((None, None))
        return res