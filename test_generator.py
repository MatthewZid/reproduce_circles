import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Concatenate, ReLU, LeakyReLU, Add
from tensorflow.python.keras.models import Model
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from utils import *
from tqdm import trange
from circle_env import CircleEnv
from sklearn.metrics import classification_report

env = CircleEnv()
load_trpo_weights = True

def create_generator(state_dims, code_dims):
    initializer = tf.keras.initializers.GlorotNormal()
    states = Input(shape=state_dims)
    x = Dense(100, kernel_initializer=initializer, activation='tanh')(states)
    # x = ReLU()(x)
    codes = Input(shape=code_dims)
    c = Dense(64, kernel_initializer=initializer, activation='tanh')(codes)
    # c = ReLU()(c)
    # h = Add()([x, c])
    h = Concatenate(axis=1)([x,c])
    actions = Dense(2)(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

def create_posterior(state_dims, action_dims, code_dims):
    initializer = tf.keras.initializers.HeNormal()
    states = Input(shape=state_dims)
    actions = Input(shape=action_dims)
    merged = tf.concat([states,actions], 1)
    x = Dense(128, kernel_initializer=initializer)(merged)
    x = LeakyReLU()(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = LeakyReLU()(x)
    x = Dense(code_dims)(x)
    output = tf.keras.activations.softmax(x)

    model = Model(inputs=[states, actions], outputs=output)
    return model

def generate_policy(generator, code):
    s_traj = []
    a_traj = []
    c_traj = []

    # generate actions for every current state
    state_obsrv = env.reset() # reset environment state
    code_tf = tf.constant(code)
    code_tf = tf.expand_dims(code_tf, axis=0)

    while True:
        # 1. generate actions with generator
        state_tf = tf.constant(state_obsrv)
        state_tf = tf.expand_dims(state_tf, axis=0)
        action_mu = generator([state_tf, code_tf], training=False)
        action_mu = tf.squeeze(action_mu).numpy()

        # current_state = (state_obsrv[-2], state_obsrv[-1])
        s_traj.append(state_obsrv)
        a_traj.append(action_mu)
        c_traj.append(code)

        # 2. environment step
        state_obsrv, done = env.step(action_mu)

        if done:
            s_traj = np.array(s_traj, dtype=np.float32)
            a_traj = np.array(a_traj, dtype=np.float32)
            c_traj = np.array(c_traj, dtype=np.float32)
            break

    return (s_traj, a_traj, c_traj)

generator = create_generator(10, 3)
posterior = create_posterior(10, 2, 3)

if load_trpo_weights: generator.load_weights('./saved_models/trpo/generator.h5')
else: generator.load_weights('./saved_models/bc/generator.h5')
posterior.load_weights('./saved_models/trpo/posterior.h5')

expert_states, expert_actions, expert_codes = pkl.load(open("expert_traj.pkl", "rb"))

expert_states = np.concatenate(expert_states)
expert_actions = np.concatenate(expert_actions)
expert_codes = np.concatenate(expert_codes)

# InfoGAIL accuracy method
sampled_expert_idx = np.random.choice(expert_states.shape[0], 2000, replace=False)
sampled_expert_states = expert_states[sampled_expert_idx, :]
sampled_expert_actions = expert_actions[sampled_expert_idx, :]
sampled_expert_codes = np.argmax(expert_codes[sampled_expert_idx, :], axis=1)
probs = posterior([sampled_expert_states, sampled_expert_actions], training=False).numpy()
codes_pred = np.argmax(probs, axis=1)

print('Posterior accuracy over expert state-action pairs')
print(classification_report(sampled_expert_codes, codes_pred))

colors = ['red','blue','green']
plt.figure()
# plt.xlim(-1,1)
# plt.ylim(-1,1)
for i in trange(8):
    pick = np.random.choice(3, 1)[0]
    one_hot = np.zeros((3,))
    one_hot[pick] = 1
    traj = generate_policy(generator, one_hot)
    plt.scatter(traj[0][:, -2], traj[0][:, -1], s=4, c=colors[pick], alpha=0.4)
plt.savefig("./plots/generated_trajectories", dpi=100)
plt.close()