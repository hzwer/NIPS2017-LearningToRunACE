# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from osim.env import RunEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import json
import multiprocessing
from time import sleep

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit

from multi import fastenv
from farmer import farmer as farmer_class
farmer = farmer_class()
graph = tf.get_default_graph()
actor_num = 1
critic_num = 1
import itertools

class Game(object):
    #Tensorflow GPU optimization
    TOTSTEP = 0
    TOTREWARD = 0.
    RUNTIME = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    action_dim = 18
    state_dim = 76
    max_steps = 1000
    cnt = 0
    falloff = 0
    actor = []
    critic = []
    mx = 0
    
    import threading as th
    lock = th.Lock()

    for i in range(actor_num):
        actor.append(ActorNetwork(sess, state_dim, action_dim, 0, 0, 0))
    for i in range(critic_num):
        critic.append(CriticNetwork(sess, state_dim, action_dim, 0, 0, 0))
    
    def play(self, env, cnt):
        step = 0        
        s_t = env.reset()
        total_reward = 0.
        sp = 0.
        for j in range(self.max_steps):
            self.lock.acquire()
            global graph
            ACT = []
            pre_reward = []
            for i in range(actor_num):
                pre_reward.append(0.)
            with graph.as_default():
                for i in range(actor_num):
                    ACT.append(self.actor[i].model.predict(np.array([s_t]))[0])
                    for j in range(critic_num):
                        pre_reward[i] += (self.critic[j].model.predict([np.array([s_t]), np.array([ACT[i]])])[0])
            self.lock.release()
            best = np.argmax(pre_reward)
            a_t = ACT[best]
            ob, r_t, done, _, pen = env.step(a_t)
            s_t1 = ob
            total_reward += r_t
            sp += pen
            s_t = s_t1
            step += 1
            if step % 10 == 0:
                print("Episode", cnt, "Step", step, "Reward", total_reward)
            if done or step == 1000 / 1:
                self.lock.acquire()
                if total_reward > self.mx:
                    self.mx = total_reward
                self.TOTSTEP = self.TOTSTEP + step
                self.TOTREWARD = self.TOTREWARD + total_reward
                if step != 1000:
                    self.falloff = self.falloff + 1
                print("Episode", cnt, "Step", step, "Reward", total_reward, "penalty", sp, "falloff", self.falloff)
                self.RUNTIME = self.RUNTIME + 1
                print("totstep", self.TOTSTEP, "totreward", self.TOTREWARD / self.RUNTIME)
                self.lock.release()
                break

    def playonce(self, env, T):
        from multi import fastenv
        fenv = fastenv(env, 1)
        self.play(fenv, T)
        env.rel()
        del fenv
    
    def play_ignore(self, env, T):
        import threading as th
        try:
            t = th.Thread(target=self.playonce,args=(env,T,))
            t.setDaemon(True)
            t.start()
        except:
            print("startfail")
    
    def playifavailable(self, T):
        while True:
            remote_env = farmer.acq_env()
            if remote_env == False:
                pass
            else:
                self.play_ignore(remote_env, T)
                break
    
    def pre(self):        
        print("Now we load the weight")
        try:
            for i in range(actor_num):                
                self.actor[i].model.load_weights("logs/actormodel{}.h5".format(i + 1))
            for i in range(critic_num):
                self.critic[i].model.load_weights("logs/criticmodel{}.h5".format(i + 1))
            print("Weight load successfully")
        except:
            return False
        finally:
            return True
    
    def run(self):
        np.random.seed(1)
        reward = 0
        done = False
        LOSS = 0
        
        for T in range(10):
            self.playifavailable(T)
        while(self.RUNTIME < 10):
            sleep(1)
        print("Finish.", self.TOTREWARD / 10)
    
if __name__ == "__main__":
    t = Game()
    if(t.pre() == False):
        exit()
    t.run()

