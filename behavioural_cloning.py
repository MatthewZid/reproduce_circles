import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Add
from tensorflow.keras.models import Model
import numpy as np
import pickle as pkl
from trpo import *

BATCH_SIZE = 128

def create_generator(state_dims, code_dims, initializer):
    states = Input(shape=state_dims)
    # x = Flatten()(states)
    x = Dense(128, kernel_initializer=initializer)(states)
    x = ReLU()(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    codes = Input(shape=code_dims)
    c = Dense(128, kernel_initializer=initializer)(codes)
    h = Add()([x, c])
    h = ReLU()(h)
    actions = Dense(2, activation='tanh')(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

def create_discriminator(state_dims, action_dims, initializer):
    states = Input(shape=state_dims)
    actions = Input(shape=action_dims)
    merged = tf.concat([states,actions], 1)
    # x = Flatten()(merged)
    x = Dense(128, kernel_initializer=initializer)(merged)
    x = ReLU()(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = ReLU()(x)
    score = Dense(1)(x)

    model = Model(inputs=[states, actions], outputs=score)
    return model

def create_posterior(state_dims, action_dims, code_dims, initializer):
    states = Input(shape=state_dims)
    actions = Input(shape=action_dims)
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

def create_valuenet(state_dims, code_dims, initializer):
    states = Input(shape=state_dims)
    codes = Input(shape=code_dims)
    merged = tf.concat([states,codes], 1)
    x = Dense(256, kernel_initializer=initializer)(merged)
    x = ReLU()(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = ReLU()(x)
    output = Dense(1)(x)

    model = Model(inputs=[states, codes], outputs=output)
    return model

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

train_ratio = int((80*expert_states.shape[0])/100.0)
train_states = shuffled_expert_states[0:train_ratio, :]
val_states = shuffled_expert_states[train_ratio:, :]
train_actions = shuffled_expert_actions[0:train_ratio, :]
val_actions = shuffled_expert_actions[train_ratio:, :]
train_codes = shuffled_expert_codes[0:train_ratio, :]
val_codes = shuffled_expert_codes[train_ratio:, :]

train_states = tf.convert_to_tensor(train_states, dtype=tf.float32)
train_actions = tf.convert_to_tensor(train_actions, dtype=tf.float32)
train_codes = tf.convert_to_tensor(train_codes, dtype=tf.float32)
val_states = tf.convert_to_tensor(val_states, dtype=tf.float32)
val_actions = tf.convert_to_tensor(val_actions, dtype=tf.float32)
val_codes = tf.convert_to_tensor(val_codes, dtype=tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((train_states, train_actions, train_codes))
train_data = train_data.batch(batch_size=BATCH_SIZE)

# train
initializer = tf.keras.initializers.HeNormal()
generator = create_generator(10, 3, initializer)
discriminator = create_discriminator(10, 2, initializer)
posterior = create_posterior(10, 2, 3, initializer)
valuenet = create_valuenet(10, 3, initializer)

gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
disc_optimizer = tf.keras.optimizers.RMSprop()
post_optimizer = tf.keras.optimizers.Adam()
mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.CategoricalCrossentropy()
actions_logstd = np.zeros((BATCH_SIZE, expert_actions.shape[1]), dtype=np.float32)

for _, (states_batch, actions_batch, codes_batch) in enumerate(train_data):
    with tf.GradientTape() as gen_tape:
        actions_mu = generator([states_batch, codes_batch], training=True)
        z = tf.random.normal([BATCH_SIZE,2], 0, 0.4)
        actions = actions_mu + tf.math.exp(actions_logstd) * z
        actions = np.clip(actions, -1, 1)
        gen_loss = mse(actions_batch, actions)
    
    policy_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
    gen_optimizer.apply_gradients(zip(policy_gradients, generator.trainable_weights))

    with tf.GradientTape() as disc_tape:
        vals = discriminator([states_batch, actions_batch], training=True)
        disc_loss = mse(tf.ones_like(vals), vals)
    
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_weights))

    with tf.GradientTape() as post_tape:
        probs = posterior([states_batch, actions_batch], training=True)
        post_loss = cross_entropy(codes_batch, probs)
    
    post_gradients = post_tape.gradient(post_loss, posterior.trainable_weights)
    post_optimizer.apply_gradients(zip(post_gradients, posterior.trainable_weights))

    # value net training???