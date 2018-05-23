import numpy as np
import math
import time

def add_advantage_macro(rewards, macro_vpred, macrolen, gamma, lam):
    # new = np.append(seg["new"][0::macrolen], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    # print ('seg[new]: ', seg["new"])
    # print ('seg[new][0::macrolen]: ', seg["new"][0::macrolen])

    vpred = np.append(macro_vpred, 0)
    # T = int(len(seg["rew"])/macrolen)
    num_macro_acts = len(macro_vpred)
    T = num_macro_acts
    # seg["macro_adv"] = gaelam = np.empty(T, 'float32')
    macro_adv = np.empty(num_macro_acts, 'float32')
    rew = rewards
    lastgaelam = 0
    for t in reversed(range(num_macro_acts)):
        # nonterminal = 1-new[t+1]
        nonterminal = 1 if (t != num_macro_acts-1) else 0
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        macro_adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    macro_tdlamret = macro_adv + macro_vpred

    # if sum(rewards) > 0:
    #     print ('rewards: ', rewards)
    #     print ('m_vpred: ', macro_vpred)
    #     print ('m_adv  : ', macro_adv)

    return macro_adv, macro_tdlamret

