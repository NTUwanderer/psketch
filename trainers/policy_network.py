from .rl_algs.common.mpi_running_mean_std import RunningMeanStd
from .rl_algs.common import tf_util as U
import tensorflow as tf
import numpy as np
# import gym
from .rl_algs.common.distributions import CategoricalPdType

class Policy(object):
    def __init__(self, name, ob, ac_space, hid_size, num_hid_layers, num_subpolicies, prev_act, prev_weight, start_counts, next_counts, gaussian_fixed_var=True):
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.num_subpolicies = num_subpolicies
        self.gaussian_fixed_var = gaussian_fixed_var
        self.num_subpolicies = num_subpolicies

        all_counts = np.append(np.reshape(start_counts, (1, -1)), next_counts, axis=0)

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1],))
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            # obz = ob

            self.all_counts = tf.constant(all_counts)
            # base = tf.reshape(tf.tile(tf.ones_like(prev_weight) - prev_weight, tf.constant(num_subpolicies, shape=[1])), (-1, num_subpolicies))
            test = tf.ones_like(prev_weight) - prev_weight
            mytile = tf.constant(num_subpolicies, shape=[1])
            print ('test: ', test)
            print ('mytile: ', mytile)
            base = tf.reshape(tf.tile(test, mytile), (-1, num_subpolicies))

            weight_base = tf.gather(self.all_counts, prev_act+tf.ones_like(prev_act))
            weight_full_dim = tf.reshape(tf.tile(prev_weight, tf.constant(num_subpolicies, shape=[1])), (-1, num_subpolicies))
            print ('weight_base: ', weight_base)
            print ('weight_full_dim: ', weight_full_dim)
            weights = weight_base * weight_full_dim
            print ("shapes: ", base, weights)
            self.weights = base + weights

            # value function
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

            # master policy
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "masterpol%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.selector = self.weights * U.dense(last_out, num_subpolicies, "masterpol_final", U.normc_initializer(0.01))
            self.pdtype = pdtype = CategoricalPdType(num_subpolicies)
            self.pd = pdtype.pdfromflat(self.selector)

        # all probs
        self._acts = U.function([ob, prev_act, prev_weight], [self.selector, self.vpred])

        # sample actions
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, prev_act, prev_weight], [ac, self.vpred])

        # debug
        self._debug = U.function([stochastic, ob], [ac, self.selector])
        self._act_forced = U.function([stochastic, ob, self.selector], [ac, self.vpred])

    def act(self, stochastic, ob, prev_act, prev_weight):
        ac1, vpred1 =  self._act(stochastic, ob[None], prev_act, prev_weight)
        return ac1[0], vpred1[0]
    def getActs(self, obs, prev_acts, prev_weights):
        acts, vpreds = self._acts(obs, prev_acts, prev_weights)
        return acts, vpreds
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

    # debug
    def act_forced(self, stochastic, ob, policy):
        softmax = np.zeros(self.num_subpolicies)
        softmax[policy] = 1
        ac1, vpred1 =  self._act_forced(stochastic, ob[None], softmax)
        return ac1[0], vpred1[0]
    def debug(self, stochastic, ob):
        _, sel = self._debug(stochastic, ob[None])
        return sel[0]
