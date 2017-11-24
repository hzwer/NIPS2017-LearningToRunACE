from multiprocessing import Process, Pipe

# FAST ENV

# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.

from observation_processor import generate_observation as go
import numpy as np

class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        self.stepcount = 0

        self.old_observation = None
        self.skipcount = skipcount # 4

    def obg(self,plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.stepcount)
        return np.array(processed_observation)

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

        sr = 0
        sp = 0
        for j in range(self.skipcount):
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)

            headx = oo[22]
            px = oo[1]
            py = oo[2]
            kneer = oo[7]
            kneel = oo[10]
            lean = min(0.3, max(0, px - headx - 0.15)) * 0.05
            joint = sum([max(0, k-0.1) for k in [kneer, kneel]]) * 0.03
            penalty = lean + joint            

            o = self.obg(oo)
            sr += r
            sp += penalty
            
            if d == True:
                break

        return o,sr,d,i,sp

    def reset(self):
        self.stepcount=0
        self.old_observation = None

        oo = self.e.reset()
        # o = self.e.reset(difficulty=2)
        self.lastx = oo[1]
        o = self.obg(oo)
        return o
