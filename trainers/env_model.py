from .rl_algs.common.mpi_running_mean_std import RunningMeanStd
from .rl_algs.common import tf_util as U
import tensorflow as tf
import numpy as np
# import gym
from .rl_algs.common.distributions import CategoricalPdType

class EnvModel(object):
    def __init__(self, name, ob, hid_size, num_hid_layers, num_subpolicies, gaussian_fixed_var=True):
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.num_subpolicies = num_subpolicies
        self.gaussian_fixed_var = gaussian_fixed_var
        self.num_subpolicies = num_subpolicies

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1],))
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            # obz = ob

            # value function
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

            # master policy
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "envmodel%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.env_pred = U.dense(last_out, ob.get_shape()[1], "envmodel_final", U.normc_initializer(1.0))

        # sample actions
        ac = tf.placeholder(dtype=tf.int8, shape=(num_subpolicies))
        self._act = U.function([ac, ob], [self.env_pred, self.vpred])

    def act(self, action, ob):
        ac = np.zeros((self.num_subpolicies))
        ac[action] = 1
        env_pred1, vpred1 =  self._act(ac[None], ob[None])
        return env_pred1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def reset(self, sess):
        with tf.variable_scope(self.scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            # U.get_session().run(initializer)
            sess.run(initializer)

