import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, ReLU, LeakyReLU, Add
from tensorflow.python.keras.models import Model
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from trpo import *
from tqdm import trange
from circle_env import CircleEnv

env = CircleEnv()
load_trpo_weights = True

def create_generator(state_dims, code_dims):
    initializer = tf.keras.initializers.HeNormal()
    states = Input(shape=state_dims)
    x = Dense(100, kernel_initializer=initializer)(states)
    x = ReLU()(x)
    codes = Input(shape=code_dims)
    c = Dense(64, kernel_initializer=initializer)(codes)
    c = ReLU()(c)
    # h = Add()([x, c])
    h = tf.concat([x,c], 1)
    actions = Dense(2)(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

def generate_policy(generator, code, starting_point):
    s_traj = []
    a_traj = []
    c_traj = []

    logstd = np.array([-0.4, -0.4])

    # generate actions for every current state
    state_obsrv = env.reset(start=starting_point) # reset environment state
    code_tf = tf.constant(code)
    code_tf = tf.expand_dims(code_tf, axis=0)

    while True:
        # 1. generate actions with generator
        state_tf = tf.constant(state_obsrv)
        state_tf = tf.expand_dims(state_tf, axis=0)
        action_mu = generator([state_tf, code_tf], training=False)
        action_mu = tf.squeeze(action_mu).numpy()

        action_std = np.exp(logstd)
        # sample action
        # z = np.random.randn(1, logstd.shape[0])
        # action = action_mu + action_std * z[0]

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

if load_trpo_weights: generator.load_weights('./saved_models/trpo/generator.h5')
else: generator.load_weights('./saved_models/bc/generator.h5')

expert_states, expert_actions, expert_codes = pkl.load(open("expert_traj.pkl", "rb"))

expert_states = np.concatenate(expert_states)
expert_actions = np.concatenate(expert_actions)
expert_codes = np.concatenate(expert_codes)

colors = ['red','blue','green']
plt.figure()
# plt.xlim(-1,1)
# plt.ylim(-1,1)
for i in trange(8):
    pick = np.random.choice(expert_codes.shape[0], 1)[0]
    traj = generate_policy(generator, expert_codes[pick], (expert_states[pick, -2], expert_states[pick, -1]))
    argcolor = np.where(expert_codes[pick] == 1)[0][0] # find the index of code from one-hot
    plt.scatter(traj[0][:, -2], traj[0][:, -1], s=4, c=colors[argcolor], alpha=0.4)
plt.savefig("./plots/generated_trajectories", dpi=100)
plt.close()