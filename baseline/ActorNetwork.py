import numpy as np
import math
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 800
HIDDEN2_UNITS = 400

def act(x):
        return 1.67653251702 * (x * K.sigmoid(x) - 0.20662096414)

def tanh(x):
        return (K.tanh(x) + 1) / 2

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = tf.placeholder(tf.float32,shape=[])

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads, learning_rate):
            
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
            self.LEARNING_RATE : learning_rate
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
    
    def create_actor_network(self, state_size,action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer = 'normal')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal')(h0)
        V = Dense(action_dim ,activation=tanh, kernel_initializer = 'normal')(h1)
        model = Model(inputs=S,outputs=V)
        model.summary()
        return model, model.trainable_weights, S
