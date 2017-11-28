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

from rpm import rpm
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit

from multi import fastenv
from farmer import farmer as farmer_class
farmer = farmer_class()
graph = tf.get_default_graph()

import itertools

class Game(object):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    mx = 0
    LOG = 0
    trainnum = 0
    modelcnt = 0
    noiselevel = 0.5
    rpm = rpm(2000000)       
    TAU = 0.001
    lr_actor = 3e-4
    lr_critic = 3e-4
    train_interval = 1
    train_times = 100
    action_dim = 18
    state_dim = 76
    max_steps = 1000 // 4 
    cnt = 0
    GAMMA = 0.96
    BATCH_SIZE = 128
    log_path = './logs'
    
    import threading as th
    lock = th.Lock()
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, lr_actor)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, lr_critic)
    
    callback = TensorBoard(log_path)
    callback.set_model(critic.model)

    def write_log(self, callback, names, logs, batch_no):
        output = open('logs/data.txt', 'w')
        output.write(str(self.LOG) + ' ' + str(self.trainnum))
        output.close()
        for name, value in zip(names, itertools.repeat(logs)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
        callback = TensorBoard(self.log_path)
    
    def play(self, env, cnt):
        episode_memory = []
        step = 0
        s_t = env.reset()
        total_reward = 0.
        sp = 0.
        noise_t = np.zeros([1, self.action_dim])
        a_t = np.zeros([1, self.action_dim])
        noise = self.noiselevel
        self.noiselevel = noise * 0.999
        for j in range(self.max_steps):
            self.lock.acquire()
            global graph
            with graph.as_default():
                a_t_original = self.actor.model.predict(np.array([s_t]))
            self.lock.release()
            noise = noise * 0.98
            if cnt % 3 == 0:
                if j % 5 == 0:
                    noise_t[0] = np.random.randn(self.action_dim) * noise
            elif cnt % 3 == 1:
                if j % 5 == 0:
                    noise_t[0] = np.random.randn(self.action_dim) * noise * 2
            else:
                noise_t = np.zeros([1, self.action_dim])
            a_t = a_t_original + noise_t
            for i in range(self.action_dim):
                if(a_t[0][i] > 1):
                    a_t[0][i] = 1
                elif(a_t[0][i] < 0):
                    a_t[0][i] = 0
            ob, r_t, done, _, pen = env.step(a_t[0])
            s_t1 = ob            
            episode_memory.append([s_t, a_t[0], r_t - pen, done, s_t1])
            total_reward += r_t
            sp += pen
            s_t = s_t1
            step += 1
            if done or step == 1000 / 4 - 1: 
                if total_reward > self.mx:
                    self.mx = total_reward
                print("Episode", cnt, "Step", step, "Reward", total_reward, "max", self.mx, "penalty", sp)
                train_names = ['reward']
                self.lock.acquire()
                self.LOG = self.LOG + 1
                self.write_log(self.callback, train_names, total_reward, self.LOG)
                self.lock.release()
                break
        self.lock.acquire()        
        for i in range(step):
            self.rpm.add(episode_memory[i])
        self.lock.release()

    def playonce(self, env, T):
        from multi import fastenv
        fenv = fastenv(env, 4)
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

    def train(self):
        memory = self.rpm
        if memory.size() < self.BATCH_SIZE:
            return
        global graph
        loss = 0
        for T in range(self.train_times):
            [states,actions,rewards,dones,new_states] = memory.sample_batch(self.BATCH_SIZE)
            y_t = np.asarray([0.0] * self.BATCH_SIZE)
            rewards = np.concatenate(rewards)
            self.lock.acquire()
            with graph.as_default():
                target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
            target_q_values = target_q_values.reshape([1,target_q_values.shape[0]])[0]
            for k in range(self.BATCH_SIZE):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.GAMMA * target_q_values[k]
            with graph.as_default():
                self.critic.model.optimizer.learning_rate = self.lr_critic
                logs = self.critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads, learning_rate = self.lr_actor)
                self.actor.target_train()
                self.critic.target_train()
            train_names = ['train_loss']
            self.write_log(self.callback, train_names, logs, self.trainnum)
            self.trainnum = self.trainnum + 1
            loss = loss + logs
            self.lock.release()
        print("train", memory.size(), loss)

    def save(self):
        self.modelcnt = self.modelcnt + 1
        self.actor.target_model.save_weights("logs/actormodel.h5", overwrite=True)
        self.critic.target_model.save_weights("logs/criticmodel.h5", overwrite=True)
        self.actor.target_model.save_weights("logs/actormodel{}.h5".format(self.modelcnt))
        self.critic.target_model.save_weights("logs/criticmodel{}.h5".format(self.modelcnt))
        print("save")
  
    def pre(self):
        print("Now we load the weight")
        try:
            input = open('logs/data.txt', 'r')
            self.LOG, self.trainnum = map(int, input.read().split(' '))
            print("LOG", self.LOG, "trainnum", self.trainnum)
            input.close()
            print("log found")
            self.critic.model.load_weights("logs/criticmodel.h5")
            self.critic.target_model.load_weights("logs/criticmodel.h5")
            self.actor.model.load_weights("logs/actormodel.h5")
            self.actor.target_model.load_weights("logs/actormodel.h5")
            print("Weight load successfully")
            self.rpm.load('logs/rpm.pickle')
            print("rmp success")
        except:
            if self.LOG > 0:
                print("Load fault")
                return False
            else:
                print("A new experiment") 
        return True

    def run(self):
        np.random.seed(23333)
        episode_count = 10000
        reward = 0
        done = False
        LOSS = 0
        
        for T in range(50):
            self.playifavailable(T)
        for T in range(episode_count):
            self.train()
            self.playifavailable(T)
            if np.mod(T, 100) == 0 and T >= 100:
                self.save()
        print("Finish.")

if __name__ == "__main__":
    t = Game()
    if(t.pre() == False):
        exit()
    t.run()
