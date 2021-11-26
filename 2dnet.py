import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Flatten, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import random

from tensorflow.python.keras.engine import training
from circle_env import CircleEnv
import tqdm
import math

cpu_device = tf.config.get_visible_devices()
tf.config.set_visible_devices(cpu_device[0], 'CPU')

class CircleAgent():
    def __init__(self, state_dims, action_dims, code_dims, batch_size=32, max_ep=1024):
        self.env = CircleEnv(max_step=200)
        self.max_ep = max_ep
        initializer = tf.keras.initializers.HeUniform()
        self.batch = batch_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.code_dims = code_dims
        self.generator = self.create_generator(initializer)
        self.discriminator = self.create_discriminator(initializer)
        print('\nAgent created')

    def create_generator(self, initializer):
        states = Input(shape=self.state_dims)
        # x = Flatten()(states)
        x = Dense(128, kernel_initializer=initializer)(states)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        codes = Input(shape=self.code_dims)
        c = Dense(128, kernel_initializer=initializer)(codes)
        h = Add()([x, c])
        h = ReLU()(h)
        actions = Dense(2)(h)

        model = Model(inputs=[states,codes], outputs=actions)
        return model

    def create_discriminator(self, initializer):
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        merged = tf.concat([states,actions], 1)
        # x = Flatten()(merged)
        x = Dense(128, kernel_initializer=initializer)(merged)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = ReLU()(x)
        score = Dense(1)(x)

        model = Model(inputs=[states, actions], outputs=score)
        return model
    
    def __generate_policy(self, code):
        s_traj = []
        a_traj = []

        # generate actions for every current state
        state_obsrv = self.env.reset() # reset environment state
        code_tf = tf.constant(code)
        code_tf = tf.expand_dims(code_tf, axis=0)

        while True:
            # 1. generate actions with generator
            state_tf = tf.constant(state_obsrv)
            state_tf = tf.expand_dims(state_tf, axis=0)
            action = self.generator([state_tf, code_tf], training=False)
            action = tf.squeeze(action).numpy()
            norm  = math.sqrt(action[0]*action[0] + action[1]*action[1])

            if norm > 1e-8:
                action = np.array([action[0] / norm, action[1] / norm], dtype=np.float32)
            else:
                action = np.array([0,0], dtype=np.float32)

            # current_state = (state_obsrv[-2], state_obsrv[-1])
            s_traj.append(state_obsrv)
            a_traj.append(action)

            # 2. environment step
            state_obsrv, done = self.env.step(action)

            if done:
                s_traj = np.array(s_traj, dtype=np.float32)
                a_traj = np.array(a_traj, dtype=np.float32)
                break
        
        return (s_traj, a_traj)
    
    def view_traj(self, traj):
        plt.scatter(traj[:,-2], traj[:,-1], c=['red'], alpha=0.4)
        plt.show()
    
    def infogail(self):
        # load data
        expert_states, expert_actions, code_prob = pkl.load(open("circle_traj.pkl", "rb"))

        # Sample a batch of latent codes: ci ∼ p(c)
        sampled_codes = np.zeros((self.batch, self.code_dims))
        code_ids = np.arange(0,self.code_dims)
        for i in range(self.batch):
            pick = np.random.choice(code_ids, p=code_prob)
            sampled_codes[i, pick] = 1
        
        # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
        traj = []
        for code in sampled_codes:
            trajectory = self.__generate_policy(code)
            traj.append(trajectory)

        # self.view_traj(traj[0][0])

        # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
        

# main
agent = CircleAgent(10, 2, 3)
agent.infogail()