import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU, Add
from tensorflow.keras.models import Model
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from trpo import *
from tqdm import trange
from circle_env import CircleEnv

BATCH_SIZE = 2048
show_fig = False
show_traj = True

def create_generator(state_dims, code_dims):
    initializer = tf.keras.initializers.GlorotUniform()
    states = Input(shape=state_dims)
    x = Dense(128, activation='tanh', kernel_initializer=initializer)(states)
    # x = ReLU()(x)
    x = Dense(128, activation='tanh', kernel_initializer=initializer)(x)
    # x = ReLU()(x)
    codes = Input(shape=code_dims)
    c = Dense(128, activation='tanh', kernel_initializer=initializer)(codes)
    # c = ReLU()(c)
    h = Add()([x, c])
    # h = tf.concat([x,c], 1)
    actions = Dense(2, activation='tanh')(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

def standardize(states):
    for i in range(states.shape[1]):
        states[:,i] = (states[:,i] - states[:,i].mean()) / states[:,i].std()

env = CircleEnv(max_step=256)

def generate_policy(generator, code):
    s_traj = []
    a_traj = []
    c_traj = []

    # logstd = np.array([0.0,0.0])

    # generate actions for every current state
    state_obsrv = env.reset() # reset environment state
    code_tf = tf.constant(code)
    code_tf = tf.expand_dims(code_tf, axis=0)

    while True:
        # 1. generate actions with generator
        state_tf = tf.constant(state_obsrv)
        state_tf = tf.expand_dims(state_tf, axis=0)
        action = generator([state_tf, code_tf], training=False)
        action = tf.squeeze(action).numpy()
        # action_std = np.exp(logstd)

        # sample action
        # z = np.random.randn(1, logstd.shape[0])
        # action = action_mu + action_std * z[0]
        # action = np.clip(action, -1, 1)

        # current_state = (state_obsrv[-2], state_obsrv[-1])
        s_traj.append(state_obsrv)
        a_traj.append(action)
        c_traj.append(code)

        # 2. environment step
        state_obsrv, done = env.step(action)

        if done:
            s_traj = np.array(s_traj, dtype=np.float32)
            a_traj = np.array(a_traj, dtype=np.float32)
            c_traj = np.array(c_traj, dtype=np.float32)
            break

    return (s_traj, a_traj, c_traj)

# load data
expert_states, expert_actions, expert_codes, code_prob = pkl.load(open("expert_traj.pkl", "rb"))

expert_states = np.concatenate(expert_states)
expert_actions = np.concatenate(expert_actions)
expert_codes = np.concatenate(expert_codes)

# pretrain with behavioural cloning
idx = np.arange(expert_states.shape[0])
np.random.shuffle(idx)
shuffled_expert_states = expert_states[idx]
shuffled_expert_actions = expert_actions[idx]
shuffled_expert_codes = expert_codes[idx]

train_ratio = int((70*expert_states.shape[0])/100.0)
train_states = shuffled_expert_states[0:train_ratio, :]
val_states = shuffled_expert_states[train_ratio:, :]
train_actions = shuffled_expert_actions[0:train_ratio, :]
val_actions = shuffled_expert_actions[train_ratio:, :]
train_codes = shuffled_expert_codes[0:train_ratio, :]
val_codes = shuffled_expert_codes[train_ratio:, :]

standardize(train_states)

train_states = tf.convert_to_tensor(train_states, dtype=tf.float32)
train_actions = tf.convert_to_tensor(train_actions, dtype=tf.float32)
train_codes = tf.convert_to_tensor(train_codes, dtype=tf.float32)
val_states = tf.convert_to_tensor(val_states, dtype=tf.float32)
val_actions = tf.convert_to_tensor(val_actions, dtype=tf.float32)
val_codes = tf.convert_to_tensor(val_codes, dtype=tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((train_states, train_actions, train_codes))
train_data = train_data.batch(batch_size=BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((val_states, val_actions, val_codes))
val_data = val_data.batch(batch_size=BATCH_SIZE)

# train
generator = create_generator(10, 3)

gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
mse = tf.keras.losses.MeanSquaredError()
# actions_logstd = np.zeros((BATCH_SIZE, expert_actions.shape[1]), dtype=np.float32)

epochs = 50
total_train_size = sum([el[0].shape[0] for el in list(train_data.as_numpy_iterator())])
total_val_size = sum([el[0].shape[0] for el in list(val_data.as_numpy_iterator())])
result_train = []
result_val = []
for epoch in trange(epochs, desc='Epoch'):
    loss = 0.0
    for _, (states_batch, actions_batch, codes_batch) in enumerate(train_data):
        with tf.GradientTape() as gen_tape:
            actions_mu = generator([states_batch, codes_batch], training=True)
            # z = tf.random.normal([BATCH_SIZE,2], 0, 0.4)
            # actions = actions_mu + tf.math.exp(actions_logstd) * z
            # actions = np.clip(actions, -1, 1)
            gen_loss = mse(actions_batch, actions_mu)
        
        policy_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(policy_gradients, generator.trainable_weights))

        loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
    
    epoch_loss = loss / total_train_size
    result_train.append(epoch_loss)

    loss = 0.0
    for _, (states_batch, actions_batch, codes_batch) in enumerate(val_data):
        actions_mu = generator([states_batch, codes_batch], training=False)
        gen_loss = mse(actions_batch, actions_mu)
        loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
    
    epoch_loss = loss / total_val_size
    result_val.append(epoch_loss)

    # generate trajectory
    if show_traj:
        colors = ['red','green','blue']
        plt.figure()
        codes = np.zeros((3, 3))
        for i in range(3):
            codes[i,i] = 1
            traj = generate_policy(generator, codes[i])
            plt.scatter(traj[0][:, -2], traj[0][:, -1], c=colors[i], alpha=0.4)
        plt.savefig("./plots/trajectories_"+str(epoch), dpi=100)
        plt.close()

if show_fig:
    epoch_space = np.arange(1, len(result_train)+1, dtype=int)
    plt.figure()
    plt.title('Behaviour Cloning')
    plt.plot(epoch_space, result_train)
    plt.plot(epoch_space, result_val)
    plt.legend(['train loss', 'validation loss'], loc="upper right")
    plt.savefig('./plots/behaviour_cloning', dpi=100)
    plt.close()

generator.save_weights('./saved_models/generator.h5')
print('\nModels saved!')