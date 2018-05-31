from copy import copy

class Path(object):
    def __init__(self, ob):
        self.acts   = []
        self.vpreds = []
        self.obs    = [ob]

    # def __init__(self, path):
    #     self.acts   = copy(path.acts)
    #     self.vpreds = copy(path.vpreds)
    #     self.obs    = copy(path.obs)

    def add(self, act, vpred, ob):
        self.acts.append(act)
        self.vpreds.append(vpred)
        self.obs.append(ob)

    def maxReward(self, index):
        maxR = -1
        if len(self.obs) <= 1:
            return maxR
        
        for ob in self.obs[1:]:
            if ob[index] > maxR:
                maxR = ob[index]

        if maxR < 0:
            maxR = -1;

        return maxR

