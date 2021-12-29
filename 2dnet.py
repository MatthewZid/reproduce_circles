import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Flatten, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import random

from tensorflow.python.keras.engine import training
from circle_env import CircleEnv
from tqdm import tqdm
import math
from trpo import *

# cpu_device = tf.config.get_visible_devices()
# tf.config.set_visible_devices(cpu_device[0], 'CPU')

class CircleAgent():
    def __init__(self, state_dims, action_dims, code_dims, batch_size=32, max_ep=1024):
        self.env = CircleEnv(max_step=256)
        self.max_ep = max_ep
        initializer = tf.keras.initializers.HeUniform()
        self.batch = batch_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.code_dims = code_dims
        self.generator = self.create_generator(initializer)
        self.discriminator = self.create_discriminator(initializer)
        self.posterior = self.create_posterior(code_dims, initializer)
        self.value_net = self.create_valuenet(initializer)
        self.sampled_states = []
        self.sampled_actions = []
        self.sampled_codes = []
        self.disc_optimizer = tf.keras.optimizers.RMSprop()
        self.posterior_optimizer = tf.keras.optimizers.Adam()
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
    
    def create_posterior(self, code_dims, initializer):
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        merged = tf.concat([states,actions], 1)
        # x = Flatten()(merged)
        x = Dense(128, kernel_initializer=initializer)(merged)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = ReLU()(x)
        x = Dense(code_dims)(x)
        output = tf.keras.activations.softmax(x)

        model = Model(inputs=[states, actions], outputs=output)
        return model

    def create_valuenet(self, initializer):
        states = Input(shape=self.state_dims)
        codes = Input(shape=self.code_dims)
        merged = tf.concat([states,codes], 1)
        x = Dense(256, kernel_initializer=initializer)(merged)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = ReLU()(x)
        output = Dense(1)(x)

        model = Model(inputs=[states, codes], outputs=output)
        return model
    
    def __generate_policy(self, code):
        s_traj = []
        a_traj = []
        c_traj = []
        reward = 0.0

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
                action = np.array([0.0,0.0], dtype=np.float32)

            # current_state = (state_obsrv[-2], state_obsrv[-1])
            s_traj.append(state_obsrv)
            a_traj.append(action)
            c_traj.append(code)

            # 2. environment step
            state_obsrv, done = self.env.step(action)

            if done:
                s_traj = np.array(s_traj, dtype=np.float32)
                a_traj = np.array(a_traj, dtype=np.float32)
                c_traj = np.array(c_traj, dtype=np.float32)
                break
        
        return (s_traj, a_traj, c_traj)
    
    def __disc_loss(self, score):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        loss = cross_entropy(tf.ones_like(score), score)

        return loss

    def __post_loss(self, prob):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy()

        loss = cross_entropy(self.sampled_codes, prob)

        return loss
    
    def view_traj(self, traj):
        plt.scatter(traj[:,-2], traj[:,-1], c=['red'], alpha=0.4)
        plt.show()

    def train(self):
        # train discriminator
        with tf.GradientTape() as disc_tape:
            score = self.discriminator([self.sampled_states, self.sampled_actions], training=True)

            disc_loss = self.__disc_loss(score)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_weights))

        # train posterior
        with tf.GradientTape() as post_tape:
            prob = self.posterior([self.sampled_states, self.sampled_actions], training=True)

            post_loss = self.__post_loss(prob)
        
        gradients_of_posterior = disc_tape.gradient(post_loss, self.posterior.trainable_weights)
        self.posterior_optimizer.apply_gradients(zip(gradients_of_posterior, self.posterior.trainable_weights))

        # train generator (TRPO)
    
    def infogail(self):
        # load data
        expert_states, expert_actions, expert_codes, code_prob = pkl.load(open("expert_traj.pkl", "rb"))

        expert_states = np.concatenate(expert_states)
        expert_actions = np.concatenate(expert_actions)
        expert_codes = np.concatenate(expert_codes)

        # probably place an epoch loop here

        # Sample a batch of latent codes: ci ∼ p(c)
        sampled_codes = np.zeros((self.batch, self.code_dims))
        code_ids = np.arange(0,self.code_dims)
        print("\nGenerating codes...")
        for i in tqdm.tqdm(range(self.batch)):
            pick = np.random.choice(code_ids, p=code_prob)
            sampled_codes[i, pick] = 1
        
        # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
        generated_states = []
        generated_actions = []
        generated_codes = []
        print("\nGenerating trajectories...")
        for i in tqdm.tqdm(range(len(sampled_codes))):
            trajectory = self.__generate_policy(sampled_codes[i])
            generated_states.append(trajectory[0])
            generated_actions.append(trajectory[1])
            generated_codes.append(trajectory[2])
            # plt.figure()
            # plt.scatter(trajectory[0][:,-2], trajectory[0][:,-1], c=['red'], alpha=0.4)
            # plt.savefig("./plots/trajectory_"+str(i), dpi=100)
            # plt.close()
        
        generated_states = np.concatenate(generated_states)
        generated_actions = np.concatenate(generated_actions)
        generated_codes = np.concatenate(generated_codes)

        # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
        print("\nSampling state-action pairs...")
        generated_idx = np.random.choice(generated_states.shape[0], int(self.batch/2), replace=False)
        expert_idx = np.random.choice(expert_states.shape[0], int(self.batch/2), replace=False)

        generated_states = generated_states[generated_idx, :]
        generated_actions = generated_actions[generated_idx, :]
        generated_codes = generated_codes[generated_idx, :]
        sampled_expert_states = expert_states[expert_idx, :]
        sampled_expert_actions = expert_actions[expert_idx, :]
        sampled_expert_codes = expert_codes[expert_idx, :]

        self.sampled_states = np.concatenate([generated_states, sampled_expert_states])
        self.sampled_actions = np.concatenate([generated_actions, sampled_expert_actions])
        self.sampled_codes = np.concatenate([generated_codes, sampled_expert_codes])

        # shuffle indices
        idx = np.arange(len(self.sampled_states))
        np.random.shuffle(idx)
        self.sampled_states = self.sampled_states[idx]
        self.sampled_actions = self.sampled_actions[idx]
        self.sampled_codes = self.sampled_codes[idx]

        print("Creating dataset...")
        self.sampled_states = tf.convert_to_tensor(self.sampled_states, dtype=tf.float32)
        self.sampled_actions = tf.convert_to_tensor(self.sampled_actions, dtype=tf.float32)
        self.sampled_codes = tf.convert_to_tensor(self.sampled_codes, dtype=tf.float32)

        self.sampled_states = tf.data.Dataset.from_tensor_slices(self.sampled_states)
        self.sampled_states = self.sampled_states.batch(batch_size=self.batch)
        self.sampled_actions = tf.data.Dataset.from_tensor_slices(self.sampled_actions)
        self.sampled_actions = self.sampled_actions.batch(batch_size=self.batch)
        self.sampled_codes = tf.data.Dataset.from_tensor_slices(self.sampled_codes)
        self.sampled_codes = self.sampled_codes.batch(batch_size=self.batch)

        # call train here

# main
agent = CircleAgent(10, 2, 3)
agent.infogail()