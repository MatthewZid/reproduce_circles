import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Add
from tensorflow.keras.models import Model
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from trpo import *
from tqdm import trange

BATCH_SIZE = 32

def create_generator(state_dims, code_dims, initializer):
    states = Input(shape=state_dims)
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

val_data = tf.data.Dataset.from_tensor_slices((val_states, val_actions, val_codes))
val_data = val_data.batch(batch_size=BATCH_SIZE)

# train
initializer = tf.keras.initializers.HeNormal()
generator = create_generator(10, 3, initializer)

gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
disc_optimizer = tf.keras.optimizers.RMSprop()
post_optimizer = tf.keras.optimizers.Adam()
mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.CategoricalCrossentropy()
# actions_logstd = np.zeros((BATCH_SIZE, expert_actions.shape[1]), dtype=np.float32)

epochs = 20
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
    
    epoch_loss = loss / sum([el[0].shape[0] for el in list(train_data.as_numpy_iterator())])
    result_train.append(epoch_loss)

    loss = 0.0
    for _, (states_batch, actions_batch, codes_batch) in enumerate(val_data):
        actions_mu = generator([states_batch, codes_batch], training=False)
        gen_loss = mse(actions_batch, actions_mu)
        loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
    
    epoch_loss = loss / sum([el[0].shape[0] for el in list(val_data.as_numpy_iterator())])
    result_val.append(epoch_loss)

epoch_space = np.arange(1, len(result_train)+1)
plt.figure()
plt.title('Behaviour Cloning')
plt.plot(epoch_space, result_train)
plt.plot(epoch_space, result_val)
plt.legend(['train loss', 'validation loss'], loc="upper right")
plt.savefig('./plots/behaviour_cloning', dpi=100)
plt.close()

generator.save_weights('./saved_models/generator.h5')
print('\nModels saved!')