from misc import util
from misc.experience import Transition, MacroTransition
from worlds.cookbook import Cookbook

from collections import defaultdict, namedtuple
import itertools
import numpy as np
import yaml
import tensorflow as tf

from .policy_network import Policy
from .env_model import EnvModel
from .path import Path
from .rl_algs.common import tf_util as U
from .rollouts import add_advantage_macro
from .learner import Learner
import logging

N_ITERS = 3000000
N_UPDATE = 100
N_BATCH = 100
N_TEST_BATCHES = 100
IMPROVEMENT_THRESHOLD = 0.8

Task = namedtuple("Task", ["goal", "steps"])

import os
import psutil
process = psutil.Process(os.getpid())


class CurriculumTrainer(object):
    def __init__(self, config, world, model):
        # load configs
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)

        # initialize randomness
        self.random = np.random.RandomState(0)

        # organize task and subtask indices
        self.train_tasks = []
        self.test_tasks = []
        if "train" in self.hints:
            train_hints = self.hints["train"]
            test_hints = self.hints["test"]
        else:
            train_hints = self.hints
            test_hints = {}
        all_hints = dict(train_hints)
        all_hints.update(test_hints)
        hint_keys = ['make[cloth]', 'make[bed]', 'make[axe]', 'make[stick]', 'make[bridge]', 'make[plank]', 'get[gold]', 'make[shears]', 'get[gem]', 'make[rope]']
        # hint_keys = ['make[cloth]', 'make[bed]', 'make[stick]', 'make[bridge]', 'make[plank]', 'get[gold]', 'make[shears]', 'get[gem]', 'make[rope]']
        # for hint_key, hint in all_hints.items():
        for hint_key in hint_keys:
            hint = all_hints[hint_key]
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]), self.cookbook.index[goal[1]])
            if config.model.use_args:
                steps = [util.parse_fexp(s) for s in hint]
                steps = [(self.subtask_index.index(a), self.cookbook.index[b])
                        for a, b in steps]
                steps = tuple(steps)
                task = Task(goal, steps)
            else:
                steps = [self.subtask_index.index(a) for a in hint]
                steps = tuple(steps)
                task = Task(goal, steps)
            if hint_key in train_hints:
                self.train_tasks.append(task)
            else:
                self.test_tasks.append(task)
            self.task_index.index(task)

            print (hint_key, hint, task)

        model.prepare(world, self)

        print ("tasks: ", self.test_tasks)

        # self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
        hid_size=32
        num_hid_layers=2
        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, model.n_features + len(self.test_tasks)])
        self.new_ob = U.get_placeholder(name="new_ob", dtype=tf.float32, shape=[None, model.n_features + len(self.test_tasks)])
        self.acts = U.get_placeholder(name="acts", dtype=tf.int32, shape=(None, ))

        self.policy = Policy(name="policy", ob=self.ob, ac_space=world.n_actions, hid_size=hid_size, num_hid_layers=num_hid_layers, num_subpolicies=len(self.subtask_index))
        self.old_policy = Policy(name="old_policy", ob=self.ob, ac_space=world.n_actions, hid_size=hid_size, num_hid_layers=num_hid_layers, num_subpolicies=len(self.subtask_index))
        self.stochastic = True

        policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        old_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_policy')
        not_init_initializers = [var.initializer  for var in policy_vars + old_policy_vars]
        model.session.run(not_init_initializers)

        self.policy.reset(model.session)
        self.old_policy.reset(model.session)

        hid_size=1024
        num_hid_layers=2
        self.env_model = EnvModel(name="env_model", ob=self.ob, acts=self.acts, hid_size=hid_size, num_hid_layers=num_hid_layers, num_subpolicies=len(self.subtask_index))
        self.old_env_model = EnvModel(name="old_env_model", ob=self.ob, acts=self.acts, hid_size=hid_size, num_hid_layers=num_hid_layers, num_subpolicies=len(self.subtask_index))
        
        with model.session.as_default() as sess:
            self.learner = Learner(self.policy, self.old_policy, self.env_model, self.old_env_model, len(self.subtask_index), None, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)

        model.load()
        # self.learner.syncMasterPolicies()
        print ("policy variables: ", self.policy.get_variables())
        print ("policy trainable variables: ", self.policy.get_trainable_variables())
        
    def do_rollout(self, model, world, possible_tasks, task_probs):
        states_before = []
        tasks = []
        goal_names = []
        goal_args = []

        # choose tasks and initialize model
        for _ in range(N_BATCH):
            task = possible_tasks[self.random.choice(
                len(possible_tasks), p=task_probs)]
            goal, _ = task
            goal_name, goal_arg = goal
            scenario = world.sample_scenario_with_goal(goal_arg)
            states_before.append(scenario.init())
            tasks.append(task)
            goal_names.append(goal_name)
            goal_args.append(goal_arg)
        model.init(states_before, tasks)
        transitions = [[] for _ in range(N_BATCH)]

        # initialize timer
        total_reward = 0.
        timer = self.config.trainer.max_timesteps
        done = [False for _ in range(N_BATCH)]

        # act!
        # mstates should be useless
        while not all(done) and timer > 0:
            mstates_before = model.get_state()
            action, terminate = model.act(states_before)
            mstates_after = model.get_state()
            states_after = [None for _ in range(N_BATCH)]
            for i in range(N_BATCH):
                if action[i] is None:
                    assert done[i]
                elif terminate[i]:
                    win = states_before[i].satisfies(goal_names[i], goal_args[i])
                    reward = 1 if win else 0
                    states_after[i] = None
                elif action[i] >= world.n_actions:
                    states_after[i] = states_before[i]
                    reward = 0
                else:
                    reward, states_after[i] = states_before[i].step(action[i])

                if not done[i]:
                    transitions[i].append(Transition(
                            states_before[i], mstates_before[i], action[i], 
                            states_after[i], mstates_after[i], reward))
                    total_reward += reward

                if terminate[i]:
                    done[i] = True

            states_before = states_after
            timer -= 1

        return transitions, total_reward / N_BATCH

    def policy_do_rollout(self, model, world, possible_tasks, task_probs):
        taskIndices = []
        states_before = []
        states_before_master = []
        tasks = []
        goal_names = []
        goal_args = []
        subPolicies = [-1] * N_BATCH
        macro_vpreds = [-1] * N_BATCH
        rewards = [0] * N_BATCH
        policies = [-1] * N_BATCH
        policy_count = [0] * N_BATCH

        targetSteps = [4, 5, 2, 6]

        # choose tasks and initialize model
        for _ in range(N_BATCH):
            taskIndex = self.random.choice(len(possible_tasks), p=task_probs)
            task = possible_tasks[taskIndex]
            taskIndices.append(taskIndex)
            goal, _ = task
            goal_name, goal_arg = goal
            scenario = world.sample_scenario_with_goal(goal_arg)
            states_before.append(scenario.init())
            tasks.append(task)
            goal_names.append(goal_name)
            goal_args.append(goal_arg)
        model.init(states_before, tasks)
        transitions = [[] for _ in range(N_BATCH)]

        states_before_master = states_before
        mstates_before = model.get_state()

        for i in range(N_BATCH):
            if np.random.uniform() < self.config.trainer.trace_prob:
                subPolicies[i], macro_vpreds[i] = self.traceBestSubPolicy(model, states_before[i], mstates_before[i], taskIndices[i],
                                                                self.config.trainer.max_policies, self.getGoalIndex(world, goal_args[i]))
            else:
                subPolicies[i], macro_vpreds[i] = self.chooseSubPolicy(model, states_before[i], mstates_before[i], taskIndices[i])

            if np.random.uniform() < self.config.trainer.random_prob:
                subPolicies[i] = np.random.randint(0, len(self.subtask_index))
            # if policy_count[i] < len(targetSteps):
            #     subPolicies[i] = targetSteps[policy_count[i]]

        # initialize timer
        total_reward = 0.
        timer = self.config.trainer.max_timesteps
        done = [False for _ in range(N_BATCH)]

        # act!
        while not all(done) and timer > 0:
            action, terminateOfCurrentSub = model.policy_act(states_before, subPolicies)
            mstates_after = model.get_state()
            states_after = [None for _ in range(N_BATCH)]
            for i in range(N_BATCH):
                shouldEnd = False
                shouldChange = False
                reward = 0

                if action[i] is None:
                    assert done[i]
                    continue
                elif terminateOfCurrentSub[i]:
                    win = states_before[i].satisfies(goal_names[i], goal_args[i])

                    # mstates_after = model.get_state() # ???

                    policy_count[i] += 1

                    if policy_count[i] < self.config.trainer.max_policies and not win:
                        reward = 0
                        states_after[i] = states_before[i]
                        shouldChange = True
                    else:
                        reward = 1 if win else 0
                        states_after[i] = None
                        shouldEnd = True

                # elif action[i] >= world.n_actions:
                #     states_after[i] = states_before[i]
                #     reward = 0
                else:
                    reward, states_after[i] = states_before[i].step(action[i])

                rewards[i] += reward
                total_reward += reward

                if shouldChange:
                    transitions[i].append(MacroTransition(
                            states_before_master[i], mstates_before[i], taskIndices[i], subPolicies[i], 
                            states_after[i], mstates_after[i], rewards[i], macro_vpreds[i]))

                    rewards[i] = 0
                    mstates_before[i] = mstates_after[i]
                    states_before_master[i] = states_after[i]
                    if np.random.uniform() < self.config.trainer.trace_prob:
                        subPolicies[i], macro_vpreds[i] = self.traceBestSubPolicy(model, states_before[i], mstates_before[i], taskIndices[i],
                                                self.config.trainer.max_policies - policy_count[i], self.getGoalIndex(world, goal_args[i]))
                    else:
                        subPolicies[i], macro_vpreds[i] = self.chooseSubPolicy(model, states_before[i], mstates_before[i], taskIndices[i])

                    if np.random.uniform() < self.config.trainer.random_prob:
                        subPolicies[i] = np.random.randint(0, len(self.subtask_index))
                    # if policy_count[i] < len(targetSteps):
                    #     subPolicies[i] = targetSteps[policy_count[i]]

                if shouldEnd and not done[i]:
                    transitions[i].append(MacroTransition(
                            states_before_master[i], mstates_before[i], taskIndices[i], subPolicies[i], 
                            states_before[i], mstates_after[i], rewards[i], macro_vpreds[i])) # states_after -> states_before
                    # ???
                    # transitions[i].append(MacroTransition(
                    #         states_before[i], mstates_before[i], action[i], 
                    #         states_after[i], mstates_after[i], rewards[i]))

                    subPolicies[i] = -1
                    done[i] = True

            states_before = states_after
            timer -= 1

        return transitions, total_reward / N_BATCH

    def retrieveGoalNum(self, ob, goal_arg, world):
        return ob[-5 - world.cookbook.n_kinds + goal_arg]

    def getGoalIndex(self, world, goal_arg):
        return -5 - world.cookbook.n_kinds + goal_arg

    def makeOb(self, model, state, mstate, taskIndex):
        taskOneHot = np.zeros((len(self.test_tasks)))
        taskOneHot[taskIndex] = 1
        feature = model.featurize(state, mstate)
        ob = np.concatenate((taskOneHot, feature))

        return ob


    def chooseSubPolicy(self, model, state, mstate, taskIndex):
        cur_subpolicy, macro_vpred = self.policy.act(self.stochastic, self.makeOb(model, state, mstate, taskIndex))
        return cur_subpolicy, macro_vpred

    def traceBestSubPolicy(self, model, state, mstate, taskIndex, depth, goalIndex):
        branches = 2

        initOb = self.makeOb(model, state, mstate, taskIndex)
        initP = Path(initOb)
        allPaths = [initP]

        for d in range(depth):
            origSize = len(allPaths)

            obs = np.array([ path.obs[-1]  for path in allPaths ])
            acts, vpreds = self.policy.getActs(obs)
            # print ("obs: ", obs)
            # print ("acts: ", acts)

            newActs = np.zeros((branches * origSize), dtype=np.int8)
            newObs  = np.concatenate((obs, obs))
            for i in range(origSize):
                for j in range(branches):
                    maxI = np.argmax(acts[i])
                    minI = np.argmin(acts[i])
                    newActs[i + j * origSize] = maxI
                    # print ("indices: ", i, maxI, minI)
                    # print ("shape: ", len(acts), len(acts[i]))
                    acts[i][maxI] = acts[i][minI] - 1

            env_preds = self.env_model.getObs(newActs, newObs)

            temp = []
            for i in range(branches):
                temp += allPaths

            allPaths = temp

            for i in range(len(allPaths)):
                allPaths[i].add(newActs[i], vpreds[i % origSize], env_preds[i])

        bestAction = -1
        bestVpred = -1
        bestResult = -1
        for path in allPaths:
            r = path.maxReward(goalIndex)
            if bestAction == -1 or bestResult < r:
                bestAction = path.acts[0]
                bestVpred = path.vpreds[0]
                bestResult = r

        return bestAction, bestVpred

    def test(self, model, world):
        possible_tasks = self.test_tasks
        task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)
        task_rewards = defaultdict(lambda: 0)
        task_counts = defaultdict(lambda: 0)
        for i in range(N_TEST_BATCHES):
            transitions, reward = self.do_rollout(model, world, possible_tasks,
                    task_probs)
            for t in transitions:
                tr = sum(tt.r for tt in t)
                task_rewards[t[0].m1.task] += tr
                task_counts[t[0].m1.task] += 1

        finalScore = 0.0
        logging.info("[TEST]")
        for i, task in enumerate(possible_tasks):
            i_task = self.task_index[task]
            score = 1. * task_rewards[i_task] / task_counts[i_task]
            finalScore += score / len(possible_tasks)
            logging.info("[task] %s[%s] %s %s", 
                    self.subtask_index.get(task.goal[0]),
                    self.cookbook.index.get(task.goal[1]),
                    task_probs[i],
                    score)

        return finalScore

    def train(self, model, world):
        model.prepare(world, self)
        #model.load()
        if self.config.trainer.use_curriculum:
            max_steps = 1
        else:
            max_steps = 100
        i_iter = 0

        task_probs = []
        while i_iter < N_ITERS:
            logging.info("[max steps] %d", max_steps)
            min_reward = np.inf

            # TODO refactor
            for _ in range(1):
                # make sure there's something of this length
                possible_tasks = self.train_tasks
                possible_tasks = [t for t in possible_tasks if len(t.steps) <= max_steps]
                if len(possible_tasks) == 0:
                    continue

                # re-initialize task probs if necessary
                if len(task_probs) != len(possible_tasks):
                    task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

                total_reward = 0.
                total_err = 0.
                count = 0.
                task_rewards = defaultdict(lambda: 0)
                task_counts = defaultdict(lambda: 0)
                for j in range(N_UPDATE):
                    err = None
                    # get enough samples for one training step
                    while err is None:
                        i_iter += N_BATCH
                        transitions, reward = self.do_rollout(model, world, 
                                possible_tasks, task_probs)
                        for t in transitions:
                            tr = sum(tt.r for tt in t)
                            task_rewards[t[0].m1.task] += tr
                            task_counts[t[0].m1.task] += 1
                        total_reward += reward
                        count += 1
                        for t in transitions:
                            model.experience(t)
                        err = model.train()
                    total_err += err

                # log
                logging.info("[step] %d", i_iter)
                scores = []
                for i, task in enumerate(possible_tasks):
                    i_task = self.task_index[task]
                    score = 1. * task_rewards[i_task] / task_counts[i_task]
                    logging.info("[task] %s[%s] %s %s", 
                            self.subtask_index.get(task.goal[0]),
                            self.cookbook.index.get(task.goal[1]),
                            task_probs[i],
                            score)
                    scores.append(score)
                logging.info("")
                logging.info("[rollout0] %s", [t.a for t in transitions[0]])
                logging.info("[rollout1] %s", [t.a for t in transitions[1]])
                logging.info("[rollout2] %s", [t.a for t in transitions[2]])
                logging.info("[reward] %s", total_reward / count)
                logging.info("[error] %s", err / N_UPDATE)
                logging.info("")
                min_reward = min(min_reward, min(scores))

                # recompute task probs
                if self.config.trainer.use_curriculum:
                    task_probs = np.zeros(len(possible_tasks))
                    for i, task in enumerate(possible_tasks):
                        i_task = self.task_index[task]
                        task_probs[i] = 1. * task_rewards[i_task] / task_counts[i_task]
                    task_probs = 1 - task_probs
                    task_probs += 0.01
                    task_probs /= task_probs.sum()

            logging.info("[min reward] %s", min_reward)
            logging.info("")
            if min_reward > self.config.trainer.improvement_threshold:
                max_steps += 1
                model.save()

    def transfer(self, model, world):
        # model.prepare(world, self)
        # model.load()
        syncRound = 100
        i_iter = 0

        task_probs = []
        while i_iter < N_ITERS:
            min_reward = np.inf

            
            # TODO refactor
            for _ in range(1):
                possible_tasks = self.test_tasks

                # re-initialize task probs if necessary
                if len(task_probs) != len(possible_tasks):
                    task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

                total_reward = 0.
                total_err = 0.
                total_env_loss = 0.
                env_loss_count = 0
                count = 0.
                task_rewards = defaultdict(lambda: 0)
                task_counts = defaultdict(lambda: 0)
                for j in range(N_UPDATE):
                    err = None
                    # get enough samples for one training step
                    if i_iter != 0:
                        self.learner.syncMasterPolicies()
                        self.learner.syncEnvModel()

                    while err is None:
                        i_iter += N_BATCH
                        transitions, reward = self.policy_do_rollout(model, world, 
                                possible_tasks, task_probs)

                        ep_lens = []
                        ep_rets = []
                        macro_obs = []
                        macro_new_obs = []
                        # macro_rets = []
                        macro_acts = []
                        macro_advs = []
                        macro_tdlamrets = []
                        trs = []
                        for t in transitions:
                            r = []
                            vpred = []
                            tr = sum(tt.r for tt in t)
                            trs.append(tr)
                            for tt in t:
                                r.append(tt.r)
                                vpred.append(tt.vpred)

                                ob = self.makeOb(model, tt.s1, tt.m1, tt.i)
                                newOb = self.makeOb(model, tt.s2, tt.m2, tt.i)
                                total_env_loss += self.env_model.getEnvLoss(tt.a, ob, newOb)
                                env_loss_count += 1

                                macro_obs.append(ob) ## state, mstate
                                macro_new_obs.append(newOb) ## state, mstate
                                macro_acts.append(tt.a)
                                # macro_rets.append(tt.r)

                            macro_adv, macro_tdlamret = add_advantage_macro(r, vpred, self.config.model.max_subtask_timesteps, 0.99, 0.98)

                            ep_lens.append(len(t))
                            ep_rets.append(sum(r))

                            macro_advs += macro_adv.tolist()
                            macro_tdlamrets += macro_tdlamret.tolist()

                            task_rewards[t[0].m1.task] += tr
                            task_counts[t[0].m1.task] += 1

                        macro_obs = np.array(macro_obs)
                        macro_new_obs = np.array(macro_new_obs)
                        # macro_rets = np.array(macro_rets)
                        macro_acts = np.array(macro_acts)
                        macro_advs = np.array(macro_advs)
                        macro_tdlamrets = np.array(macro_tdlamrets)

                        gmean, lmean = self.learner.updateMasterPolicy(ep_lens, ep_rets, macro_obs, macro_acts, macro_advs, macro_tdlamrets)
                        self.learner.updateEnvModel(macro_obs, macro_new_obs, macro_acts)

                        total_reward += reward
                        count += 1
                        # for t in transitions:
                        #     model.experience(t)
                        # err = model.train()
                        err = 0
                    total_err += err

                    # gmean, lmean = learner.updateMasterPolicy(rolls)

                # log
                logging.info("[step] %d", i_iter)
                scores = []
                for i, task in enumerate(possible_tasks):
                    i_task = self.task_index[task]
                    score = 1. * task_rewards[i_task] / task_counts[i_task]
                    logging.info("[task] %s[%s] %s %s", 
                            self.subtask_index.get(task.goal[0]),
                            self.cookbook.index.get(task.goal[1]),
                            task_probs[i],
                            score)
                    scores.append(score)
                logging.info("")
                # logging.info("[rollout0] %s", [t.m1.action for t in transitions[0]])
                # logging.info("[rollout1] %s", [t.m1.action for t in transitions[1]])
                # logging.info("[rollout2] %s", [t.m1.action for t in transitions[2]])
                for i, transition in enumerate(transitions):
                    logging.info("[rollout %s] %s, reward: %s", str(i), [t.a for t in transition], str(trs[i]))
                    
                # logging.info("[rollout0] %s", [t.a for t in transitions[0]])
                # logging.info("[rollout1] %s", [t.a for t in transitions[1]])
                # logging.info("[rollout2] %s", [t.a for t in transitions[2]])
                logging.info("[reward] %s", total_reward / count)
                logging.info("[error] %s", err / N_UPDATE)
                logging.info("[env_loss] %s", total_env_loss / env_loss_count)
                logging.info("")
                min_reward = min(min_reward, min(scores))

            logging.info("[min reward] %s", min_reward)
            logging.info("")

            model.save()
