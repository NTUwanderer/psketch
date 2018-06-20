from .rl_algs.common.mpi_running_mean_std import RunningMeanStd
from .rl_algs.common import tf_util as U
import tensorflow as tf
import numpy as np
# import gym
from .rl_algs.common.distributions import CategoricalPdType

class EnvModel(object):
    def __init__(self, name, ob, acts, hid_size, num_hid_layers, num_subpolicies, gaussian_fixed_var=True):
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.num_subpolicies = num_subpolicies
        self.gaussian_fixed_var = gaussian_fixed_var
        self.num_subpolicies = num_subpolicies

        # acts = tf.placeholder(dtype=tf.int8, shape=(None))

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1],))
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            # obz = ob

            onehot = tf.one_hot(acts, num_subpolicies)

            # master policy
            last_out = tf.concat([obz, tf.cast(onehot, dtype=tf.float32)], axis=1)
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "envmodel%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.env_pred = U.dense(last_out, ob.get_shape()[1], "envmodel_final", U.normc_initializer(1.0))

        self._act = U.function([acts, ob], [self.env_pred])

    def act(self, action, ob):
        env_pred1 = self._act(action[None], ob[None])
        return env_pred1[0]
    def getObs(self, actions, obs):
        env_preds = self._act(actions, obs)[0]
        return env_preds
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
    def getEnvLoss(self, ob, newOb):
        return np.sum(np.square(ob - newOb))


