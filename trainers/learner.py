import numpy as np
import tensorflow as tf
from .rl_algs.common import explained_variance, fmt_row, zipsame
from .rl_algs import logger
from .rl_algs.common import tf_util as U
import time
from .rl_algs.common.mpi_adam import MpiAdam
from mpi4py import MPI
from collections import deque
from .dataset import Dataset

class Learner:
    # TODO
    def __init__(self, policy, old_policy, env_model, old_env_model, num_subpolicies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64):
        self.policy = policy
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.num_subpolicies = num_subpolicies
        # self.num_subpolicies = len(sub_policies)
        # self.sub_policies = sub_policies
        # ob_space = env.observation_space
        # ac_space = env.action_space

        # for training theta
        # inputs for training theta
        ob = U.get_placeholder_cached(name="ob")
        new_ob = U.get_placeholder_cached(name="new_ob")
        ac = policy.pdtype.sample_placeholder([None])
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        total_loss = self.policy_loss(policy, old_policy, ob, ac, atarg, ret, clip_param)
        self.master_policy_var_list = policy.get_trainable_variables()
        self.master_loss = U.function([ob, ac, atarg, ret], U.flatgrad(total_loss, self.master_policy_var_list))
        self.master_adam = MpiAdam(self.master_policy_var_list, comm=comm)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(old_policy.get_variables(), policy.get_variables())])

        env_total_loss = self.env_model_loss(env_model, old_env_model, new_ob, ac, ret, clip_param)
        self.env_model_var_list = env_model.get_trainable_variables()
        self.env_model_loss = U.function([new_ob, ac, ret], U.flatgrad(env_total_loss, self.env_model_var_list))
        self.env_model_adam = MpiAdam(self.env_model_var_list, comm=comm)

        self.assign_env_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(old_env_model.get_variables(), env_model.get_variables())])

        U.initialize()

        self.syncMasterPolicies()

    def policy_loss(self, pi, oldpi, ob, ac, atarg, ret, clip_param):
        ratio = tf.exp(pi.pd.logp(ac) - tf.clip_by_value(oldpi.pd.logp(ac), -20, 20)) # advantage * pnew / pold
        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        pol_surr = - U.mean(tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2))
        total_loss = pol_surr + vf_loss
        return total_loss

    def env_model_loss(self, m, old_m, new_ob, ac, ret, clip_param):
        vfloss1 = tf.square(m.vpred - ret)
        vpredclipped = m.vpred + tf.clip_by_value(m.vpred - old_m.vpred, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2))

        env_loss = tf.reduce_sum(tf.square(m.env_pred - new_ob))

        env_model_loss = vf_loss + env_loss
        return env_model_loss

    def syncMasterPolicies(self):
        self.master_adam.sync()

    def syncEnvModel(self):
        self.env_model_adam.sync()

    def updateMasterPolicy(self, ep_lens, ep_rets, ob, ac, atarg, tdlamret):
        #ob, ac, atarg, tdlamret = seg["macro_ob"], seg["macro_ac"], seg["macro_adv"], seg["macro_tdlamret"]
        # ob = np.ones_like(ob)
        mean = atarg.mean()
        std = atarg.std()
        meanlist = MPI.COMM_WORLD.allgather(mean)
        global_mean = np.mean(meanlist)

        real_var = std**2 + (mean - global_mean)**2
        variance_list = MPI.COMM_WORLD.allgather(real_var)
        global_std = np.sqrt(np.mean(variance_list))

        atarg = (atarg - global_mean) / max(global_std, 0.000001)

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
        optim_batchsize = min(self.optim_batchsize,ob.shape[0])

        self.policy.ob_rms.update(ob) # update running mean/std for policy

        self.assign_old_eq_new()
        for _ in range(self.optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                g = self.master_loss(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])
                self.master_adam.update(g, 0.01, 1)

        # lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        lrlocal = (ep_lens, ep_rets)
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        logger.record_tabular("EpRewMean", np.mean(rews))

        return np.mean(rews), np.mean(ep_rets)

    def updateEnvModel(self, ep_rets, new_ob, ac):

        d = Dataset(dict(new_ob=ob, ac=ac, ret=ep_rets), shuffle=True)
        optim_batchsize = min(self.optim_batchsize,ob.shape[0])

        self.env_model.ob_rms.update(ob) # update running mean/std for policy

        self.assign_env_old_eq_new()
        for _ in range(self.optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                g = self.env_model_loss(batch["new_ob"], batch["ac"], batch["ret"])
                self.env_model_adam.update(g, 0.01, 1)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
