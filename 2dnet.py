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
        self.generated_states = []
        self.generated_actions = []
        self.expert_states = []
        self.expert_actions = []
        self.sampled_expert_states = []
        self.sampled_expert_actions = []
        self.disc_optimizer = tf.keras.optimizers.RMSprop()
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
    
    def create_posterior(self, initializer):
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
    
    def __disc_loss(self, expert_score, gen_score):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        expert_loss = cross_entropy(tf.ones_like(expert_score), expert_score)
        gen_loss = cross_entropy(tf.zeros_like(gen_score), gen_score)
        total_loss = expert_loss + gen_loss

        return total_loss
    
    def view_traj(self, traj):
        plt.scatter(traj[:,-2], traj[:,-1], c=['red'], alpha=0.4)
        plt.show()

    def train(self):
        # train discriminator
        with tf.GradientTape() as disc_tape:
            expert_score = self.discriminator([self.expert_states, self.expert_actions], training=True)
            generated_score = self.discriminator([self.generated_states, self.generated_actions], training=True)

            disc_loss = self.__disc_loss(expert_score, generated_score)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # train posterior
    
    def infogail(self):
        # load data
        self.expert_states, self.expert_actions, code_prob = pkl.load(open("expert_traj.pkl", "rb"))

        self.expert_states = np.concatenate(self.expert_states)
        self.expert_actions = np.concatenate(self.expert_actions)

        # probably place an epoch loop here

        # Sample a batch of latent codes: ci ∼ p(c)
        sampled_codes = np.zeros((self.batch, self.code_dims))
        code_ids = np.arange(0,self.code_dims)
        print("\nGenerating codes...")
        for i in tqdm(range(self.batch)):
            pick = np.random.choice(code_ids, p=code_prob)
            sampled_codes[i, pick] = 1
        
        # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
        self.generated_states = []
        self.generated_actions = []
        print("\nGenerating trajectories...")
        for i in tqdm(range(len(sampled_codes))):
            trajectory = self.__generate_policy(sampled_codes[i])
            self.generated_states.append(trajectory[0])
            self.generated_actions.append(trajectory[1])
            # plt.figure()
            # plt.scatter(trajectory[0][:,-2], trajectory[0][:,-1], c=['red'], alpha=0.4)
            # plt.savefig("./plots/trajectory_"+str(i), dpi=100)
            # plt.close()
        
        self.generated_states = np.concatenate(self.generated_states)
        self.generated_actions = np.concatenate(self.generated_actions)

        # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
        print("\nSampling state-action pairs...")
        generated_idx = np.random.choice(self.generated_states.shape[0], self.batch, replace=False)
        expert_idx = np.random.choice(self.expert_states.shape[0], self.batch, replace=False)

        self.generated_states = self.generated_states[generated_idx, :]
        self.generated_actions = self.generated_actions[generated_idx, :]
        self.sampled_expert_states = self.expert_states[expert_idx, :]
        self.sampled_expert_actions = self.expert_actions[expert_idx, :]

        print("\nCreating dataset...")
        self.generated_states = tf.convert_to_tensor(self.generated_states, dtype=tf.float32)
        self.generated_actions = tf.convert_to_tensor(self.generated_actions, dtype=tf.float32)
        self.sampled_expert_states = tf.convert_to_tensor(self.sampled_expert_states, dtype=tf.float32)
        self.sampled_expert_actions = tf.convert_to_tensor(self.sampled_expert_actions, dtype=tf.float32)

        self.generated_states = tf.data.Dataset.from_tensor_slices(self.generated_states)
        self.generated_states = self.generated_states.shuffle(buffer_size=100).batch(batch_size=self.batch)
        self.generated_actions = tf.data.Dataset.from_tensor_slices(self.generated_actions)
        self.generated_actions = self.generated_actions.shuffle(buffer_size=100).batch(batch_size=self.batch)

        self.sampled_expert_states = tf.data.Dataset.from_tensor_slices(self.sampled_expert_states)
        self.sampled_expert_states = self.sampled_expert_states.shuffle(buffer_size=100).batch(batch_size=self.batch)
        self.sampled_expert_actions = tf.data.Dataset.from_tensor_slices(self.sampled_expert_actions)
        self.sampled_expert_actions = self.sampled_expert_actions.shuffle(buffer_size=100).batch(batch_size=self.batch)

        # call train here

# main
agent = CircleAgent(10, 2, 3)
agent.infogail()