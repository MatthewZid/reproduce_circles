import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, ReLU, LeakyReLU, Flatten, Add
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import random
import yaml

from circle_env import CircleEnv
from tqdm import trange
from trpo import *
from scipy.ndimage import shift

# tf.config.threading.set_inter_op_parallelism_threads(16) 
# tf.config.threading.set_intra_op_parallelism_threads(16)
# tf.config.set_soft_device_placement(True)

save_loss = True
save_models = True
resume_training = False

class CircleAgent():
    def __init__(self, state_dims, action_dims, code_dims, episodes=2000, batch_size=2048, code_batch=192, gamma=0.95, lam=0.97, max_kl=0.01):
        self.env = CircleEnv()
        self.episodes = episodes
        self.batch = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.code_batch = code_batch
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.code_dims = code_dims
        self.starting_episode = 0
        self.gen_result = []
        self.disc_result = []
        self.post_result = []
        self.value_result = []

        self.generator = self.create_generator()
        # self.old_generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.posterior = self.create_posterior(code_dims)
        self.value_net = self.create_valuenet()

        generator_weight_path = ''
        # old_generator_weight_path = ''
        if resume_training:
            with open("./saved_models/trpo/model.yml", 'r') as f:
                data = yaml.safe_load(f)
                self.starting_episode = data['episode']
                self.gen_result = data['gen_loss']
                self.disc_result = data['disc_loss']
                self.post_result = data['post_loss']
                self.value_result = data['value_loss']
            generator_weight_path = './saved_models/trpo/generator.h5'
            self.discriminator.load_weights('./saved_models/trpo/discriminator.h5')
            self.posterior.load_weights('./saved_models/trpo/posterior.h5')
            self.value_net.load_weights('./saved_models/trpo/value_net.h5')
            # old_generator_weight_path = './saved_models/trpo/old_generator.h5'
        else:
            generator_weight_path = './saved_models/bc/generator.h5'
            # old_generator_weight_path = generator_weight_path
        
        self.generator.load_weights(generator_weight_path)
        # self.old_generator.load_weights(old_generator_weight_path)
        self.expert_states = []
        self.expert_actions = []
        self.expert_codes = []
        self.trajectories = []
        
        z = np.random.randn(batch_size, action_dims)
        self.old_actions = tf.constant(z, dtype=tf.float32)

        self.disc_optimizer = tf.keras.optimizers.Adam()
        self.posterior_optimizer = tf.keras.optimizers.Adam()
        self.value_optimizer = tf.keras.optimizers.Adam()
        print('\nAgent created')

    def create_generator(self):
        initializer = tf.keras.initializers.RandomNormal()
        states = Input(shape=self.state_dims)
        x = Dense(128, kernel_initializer=initializer)(states)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU()(x)
        codes = Input(shape=self.code_dims)
        c = Dense(128, kernel_initializer=initializer)(codes)
        c = LeakyReLU()(c)
        h = Add()([x, c])
        # h = tf.concat([x,c], 1)
        actions = Dense(2)(h)

        model = Model(inputs=[states,codes], outputs=actions)
        return model

    def create_discriminator(self):
        initializer = tf.keras.initializers.RandomNormal()
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        merged = tf.concat([states,actions], 1)
        # x = Flatten()(merged)
        x = Dense(128, kernel_initializer=initializer)(merged)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU()(x)
        score = Dense(1)(x)

        model = Model(inputs=[states, actions], outputs=score)
        return model
    
    def create_posterior(self, code_dims):
        initializer = tf.keras.initializers.RandomNormal()
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        merged = tf.concat([states,actions], 1)
        # x = Flatten()(merged)
        x = Dense(128, kernel_initializer=initializer)(merged)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU()(x)
        x = Dense(code_dims)(x)
        output = tf.keras.activations.softmax(x)

        model = Model(inputs=[states, actions], outputs=output)
        return model

    def create_valuenet(self):
        initializer = tf.keras.initializers.RandomNormal()
        states = Input(shape=self.state_dims)
        codes = Input(shape=self.code_dims)
        merged = tf.concat([states,codes], 1)
        x = Dense(128, kernel_initializer=initializer)(merged)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU()(x)
        output = Dense(1)(x)

        model = Model(inputs=[states, codes], outputs=output)
        return model
    
    def __generate_policy(self, code):
        s_traj = []
        a_traj = []
        c_traj = []

        logstd = np.array([-0.4, -0.4])

        # generate actions for every current state
        state_obsrv = self.env.reset() # reset environment state
        code_tf = tf.constant(code)
        code_tf = tf.expand_dims(code_tf, axis=0)

        while True:
            # 1. generate actions with generator
            state_tf = tf.constant(state_obsrv)
            state_tf = tf.expand_dims(state_tf, axis=0)
            action_mu = self.generator([state_tf, code_tf], training=False)
            # action = self.generator([state_tf, code_tf], training=False)
            action_mu = tf.squeeze(action_mu).numpy()
            # action = tf.squeeze(action).numpy()
            action_std = np.exp(logstd)

            # sample action
            z = np.random.randn(1, logstd.shape[0])
            action = action_mu + action_std * z[0]
            # action = np.clip(action, -1, 1)

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
    
    def __disc_loss(self, score1, score2):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        genloss = cross_entropy(tf.ones_like(score1), score1)
        expertloss = cross_entropy(tf.zeros_like(score2), score2)
        # loss = tf.reduce_mean(genloss) + tf.reduce_mean(expertloss)
        loss = genloss + expertloss

        return loss

    def __post_loss(self, prob, codes_batch):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy()

        loss = cross_entropy(codes_batch, prob)
        loss = loss

        return loss
    
    def __value_loss(self, pred, returns):
        mse = tf.keras.losses.MeanSquaredError()

        loss = mse(returns, pred)

        return loss
    
    def __generator_loss(self, feed):
        # calculate ratio between old and new policy (surrogate loss)

        with tf.GradientTape() as grad_tape:
            # actions = self.generator([feed['states'], feed['codes']], training=True)
            actions_mu = self.generator([feed['states'], feed['codes']], training=True)
            actions_mu = tf.squeeze(actions_mu)
            actions_logstd = tf.ones_like(actions_mu, dtype=tf.float32) * (-0.4)
            # actions = tf.squeeze(actions)

            log_p_n = gauss_log_prob(actions_mu, actions_logstd, feed['actions'])
            log_oldp_n = gauss_log_prob(feed['old_actions'], actions_logstd, feed['actions'])
            # log_p_n = gauss_log_prob(tf.reduce_mean(actions, axis=0), tf.math.log(tf.math.reduce_std(actions, axis=0)), feed['actions'])
            # log_oldp_n = gauss_log_prob(tf.reduce_mean(feed['old_actions'], axis=0), tf.math.log(tf.math.reduce_std(feed['old_actions'], axis=0)), feed['actions'])
            # logstd = tf.constant([-0.4, -0.4], dtype=tf.float32)
            # log_p_n = gauss_log_prob(tf.reduce_mean(actions, axis=0), logstd, feed['actions'])
            # log_oldp_n = gauss_log_prob(tf.reduce_mean(feed['old_actions'], axis=0), logstd, feed['actions'])

            ratio_n = tf.exp(log_p_n - log_oldp_n)
            surrogate_loss = tf.reduce_mean(ratio_n * feed['advants'])
        
        return ((surrogate_loss, grad_tape))

    def get_loss(self, theta, feed):
        # set_from_flat(self.generator, theta)
        var_list = self.generator.trainable_weights
        shapes = [v.shape for v in var_list]
        start = 0

        weight_idx = 0
        for shape in shapes:
            size = np.prod(shape)
            self.generator.trainable_weights[weight_idx].assign(tf.reshape(theta[start:start + size], shape))
            weight_idx += 1
            start += size
        return self.__generator_loss(feed)
    
    def fisher_vector_product(self, p, feed, cg_damping=0.1):
        N = feed['states'].shape[0]
        Nf = tf.cast(N, tf.float32)
        var_list = self.generator.trainable_weights

        start = 0
        tangents = []
        for v in var_list:
            size = np.prod(v.shape)
            param = tf.reshape(p[start:(start+size)], v.shape)
            tangents.append(param)
            start += size

        with tf.GradientTape() as tape_gvp:
            tape_gvp.watch(var_list)
            with tf.GradientTape() as grad_tape:
                actions_mu = self.generator([feed['states'], feed['codes']], training=True)
                actions_logstd = tf.ones_like(actions_mu, dtype=tf.float32) * (-0.4)
                # actions = self.generator([feed['states'], feed['codes']], training=True)
                kl_firstfixed = gauss_selfKL_firstfixed(actions_mu, actions_logstd) / Nf
                # logstd = tf.constant([-0.4, -0.4], dtype=tf.float32)
                # kl_firstfixed = gauss_selfKL_firstfixed(tf.reduce_mean(actions, axis=0), logstd) / Nf

            grads = grad_tape.gradient(kl_firstfixed, var_list)
            gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]

        fvp = flatgrad(self.generator, gvp, tape_gvp) # nan gradients here!!!
        nans = tf.math.is_nan(fvp)
        if(tf.where(nans).numpy().flatten().shape[0] != 0): print('Fisher vector: NAN!!!!!!!!!!!')

        return fvp + p * cg_damping
    
    def __saveplot(self, x, y, episode, element='element'):
        plt.figure()
        plt.scatter(x, y, alpha=0.4)
        plt.savefig('./plots/'+element+'_'+str(episode), dpi=100)
        plt.close()
    
    def normalize(self, states):
        for i in range(states.shape[1]):
            states[:,i] = (states[:,i] - states[:,i].min()) / (states[:,i].max() - states[:,i].min())

    def standardize(self, states):
        for i in range(states.shape[1]):
            states[:,i] = (states[:,i] - states[:,i].mean()) / states[:,i].std()
    
    def show_loss(self):
        epoch_space = np.arange(1, len(self.gen_result)+1, dtype=int)
        plt.figure()
        plt.title('Losses')
        plt.plot(epoch_space, self.gen_result)
        plt.plot(epoch_space, self.disc_result)
        plt.plot(epoch_space, self.post_result)
        plt.legend(['gen', 'disc', 'post'], loc="lower left")
        plt.savefig('./plots/trpo_loss', dpi=100)
        plt.close()

        plt.figure()
        plt.title('Value loss')
        plt.plot(epoch_space, self.value_result)
        plt.savefig('./plots/value_loss', dpi=100)
        plt.close()

    # def __train(self, episode):
    #     # Create expert dataset
    #     expert_idx = np.arange(self.expert_states.shape[0])
    #     np.random.shuffle(expert_idx)
    #     sampled_expert_states = self.expert_states[expert_idx, :]
    #     sampled_expert_actions = self.expert_actions[expert_idx, :]
    #     sampled_expert_codes = self.expert_codes[expert_idx, :]
    #     sampled_expert_states = tf.convert_to_tensor(sampled_expert_states, dtype=tf.float32)
    #     sampled_expert_actions = tf.convert_to_tensor(sampled_expert_actions, dtype=tf.float32)
    #     sampled_expert_codes = tf.convert_to_tensor(sampled_expert_codes, dtype=tf.float32)
    #     dataset = tf.data.Dataset.from_tensor_slices((sampled_expert_states, sampled_expert_actions, sampled_expert_codes))
    #     dataset = dataset.batch(batch_size=self.batch)

    #     # Define train params
    #     aggr_disc_loss = 0.0
    #     aggr_post_loss = 0.0
    #     aggr_value_loss = 0.0
    #     aggr_gen_loss = 0.0
    #     total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])

    #     # Start batch training
    #     for _, (expert_states_batch, expert_actions_batch, expert_codes_batch) in enumerate(dataset):
    #         # Sample a batch of latent codes: ci ∼ p(c)
    #         sampled_codes = np.zeros((self.code_batch, self.code_dims))
    #         for i in range(self.code_batch):
    #             pick = np.random.choice(self.code_dims, 1)[0]
    #             sampled_codes[i, pick] = 1
            
    #         # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
    #         trajectories = []

    #         for i in range(len(sampled_codes)):
    #             trajectory_dict = {}
    #             trajectory = self.__generate_policy(sampled_codes[i])
    #             trajectory_dict['states'] = np.copy(trajectory[0])
    #             trajectory_dict['actions'] = np.copy(trajectory[1])
    #             trajectory_dict['codes'] = np.copy(trajectory[2])
    #             trajectories.append(trajectory_dict)
            
    #         generated_states = np.concatenate([traj['states'] for traj in trajectories])
    #         generated_actions = np.concatenate([traj['actions'] for traj in trajectories])
    #         generated_codes = np.concatenate([traj['codes'] for traj in trajectories])

    #         # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
    #         # generated_idx = np.random.choice(generated_states.shape[0], expert_states_batch.numpy().shape[0], replace=False)
    #         # generated_states_batch = generated_states[generated_idx, :]
    #         # generated_actions_batch = generated_actions[generated_idx, :]
    #         generated_states_batch = tf.convert_to_tensor(generated_states, dtype=tf.float32)
    #         generated_actions_batch = tf.convert_to_tensor(generated_actions, dtype=tf.float32)

    #         # train discriminator
    #         with tf.GradientTape() as disc_tape:
    #             score1 = self.discriminator([generated_states_batch, generated_actions_batch], training=True)
    #             score2 = self.discriminator([expert_states_batch, expert_actions_batch], training=True)

    #             # wasserstein loss: D(x) - D(G(z))
    #             # score1 = tf.reduce_mean(score1)
    #             # score2 = -score2
    #             # score2 = tf.reduce_mean(score2)
    #             # disc_loss = tf.math.add(score1, score2)

    #             # cross entropy loss: D(G(z)) + (1 - D(x))
    #             disc_loss = self.__disc_loss(score1, score2)
            
    #         if save_loss: aggr_disc_loss += tf.get_static_value(disc_loss) * generated_states_batch.shape[0]
    #         gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
    #         self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_weights))

    #         # train posterior
    #         # generated_codes_batch = generated_codes[generated_idx, :]
    #         generated_codes_batch = tf.convert_to_tensor(generated_codes, dtype=tf.float32)

    #         with tf.GradientTape() as post_tape:
    #             prob = self.posterior([generated_states_batch, generated_actions_batch], training=True)

    #             post_loss = self.__post_loss(prob, generated_codes_batch)
            
    #         if save_loss: aggr_post_loss += tf.get_static_value(post_loss) * generated_states_batch.shape[0]
    #         gradients_of_posterior = post_tape.gradient(post_loss, self.posterior.trainable_weights)
    #         self.posterior_optimizer.apply_gradients(zip(gradients_of_posterior, self.posterior.trainable_weights))

    #         # TRPO
    #         # calculate rewards from discriminator and posterior
    #         for traj in trajectories:
    #             reward_d = (-tf.math.log(tf.keras.activations.sigmoid(self.discriminator([traj['states'], traj['actions']], training=False)))).numpy()
    #             reward_p = tf.math.log(self.posterior([traj['states'], traj['actions']], training=False)).numpy()

    #             traj['rewards'] = reward_d.flatten() + np.sum(reward_p * traj['codes'], axis=1)

    #             # calculate values, advants and returns
    #             values = self.value_net([traj['states'], traj['codes']], training=False).numpy().flatten() # Value function
    #             baselines = np.append(values, values[-1])
    #             deltas = traj['rewards'] + self.gamma * baselines[1:] - baselines[:-1] # Advantage(st,at) = rt+1 + γ*V(st+1) - V(st)
    #             traj['advants'] = discount(deltas, self.gamma * self.lam)
    #             traj['returns'] = discount(traj['rewards'], self.gamma)
            
    #         advants = np.concatenate([traj['advants'] for traj in trajectories])
    #         advants /= (advants.std() + 1e-8)

    #         # train value net for next iter
    #         returns = np.concatenate([traj['returns'] for traj in trajectories])

    #         # returns_batch = returns[generated_idx]
    #         returns_batch = tf.convert_to_tensor(returns, dtype=tf.float32)

    #         with tf.GradientTape() as value_tape:
    #             value_pred = self.value_net([generated_states_batch, generated_codes_batch], training=True)

    #             value_loss = self.__value_loss(value_pred, returns_batch)
            
    #         if save_loss: aggr_value_loss += tf.get_static_value(value_loss) * generated_states_batch.shape[0]
    #         value_grads = value_tape.gradient(value_loss, self.value_net.trainable_weights)
    #         self.value_optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_weights))

    #         # generator training
    #         # advants_batch = advants[generated_idx]
    #         advants_batch = tf.convert_to_tensor(advants, dtype=tf.float32)

    #         # calculate previous theta (θold) and old actions
    #         thprev = get_flat(self.generator)
    #         oldactions_batch = self.generator([generated_states_batch, generated_codes_batch], training=False)
    #         # oldactions_batch = self.old_actions

    #         feed = {
    #             'states': generated_states_batch,
    #             'actions': generated_actions_batch,
    #             'codes': generated_codes_batch,
    #             'advants': advants_batch,
    #             'old_actions': oldactions_batch
    #         }

    #         (surrogate_loss, grad_tape) = self.__generator_loss(feed)
    #         if save_loss:
    #             aggr_gen_loss += tf.get_static_value(surrogate_loss) * generated_states_batch.shape[0]

    #         policy_gradient = flatgrad(self.generator, surrogate_loss, grad_tape)
    #         nans = tf.math.is_nan(policy_gradient)
    #         if(tf.where(nans).shape[0] != 0): print('NAN!!!!!!!!!!!')
    #         stepdir = conjugate_gradient(self.fisher_vector_product, feed, -policy_gradient.numpy()) # nan values here (inside fisher vector func)
    #         shs = stepdir.dot(self.fisher_vector_product(stepdir, feed)) # ...and here
    #         assert shs > 0

    #         lm = np.sqrt(shs / self.max_kl)
    #         fullstep = stepdir / lm
    #         neggdotstepdir = -policy_gradient.numpy().dot(stepdir)

    #         theta = linesearch(self.get_loss, thprev, feed, fullstep, neggdotstepdir / lm)
    #         # set_from_flat(self.generator, theta)
    #         var_list = self.generator.trainable_weights
    #         shapes = [v.shape for v in var_list]
    #         start = 0

    #         weight_idx = 0
    #         for shape in shapes:
    #             size = np.prod(shape)
    #             self.generator.trainable_weights[weight_idx].assign(tf.reshape(theta[start:start + size], shape))
    #             weight_idx += 1
    #             start += size

    #     if save_loss:
    #         episode_disc_loss = (aggr_disc_loss / total_train_size).item()
    #         episode_post_loss = (aggr_post_loss / total_train_size).item()
    #         episode_value_loss = (aggr_value_loss / total_train_size).item()
    #         episode_gen_loss = (aggr_gen_loss / total_train_size).item()
    #         self.disc_result.append(episode_disc_loss)
    #         self.post_result.append(episode_post_loss)
    #         self.value_result.append(episode_value_loss)
    #         self.gen_result.append(episode_gen_loss)
    #         if episode % 2 == 0 and episode != 0: self.show_loss()

    #     if save_models:
    #         self.generator.save_weights('./saved_models/trpo/generator.h5')
    #         self.discriminator.save_weights('./saved_models/trpo/discriminator.h5')
    #         self.posterior.save_weights('./saved_models/trpo/posterior.h5')
    #         self.value_net.save_weights('./saved_models/trpo/value_net.h5')
    #         self.old_generator.save_weights('./saved_models/trpo/old_generator.h5')
    #         yaml_conf = {
    #             'episode': episode+1,
    #             'gen_loss': self.gen_result,
    #             'disc_loss': self.disc_result,
    #             'post_loss': self.post_result,
    #             'value_loss': self.value_result
    #         }
            
    #         with open("./saved_models/trpo/model.yml", 'w') as f:
    #             yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)

    ############################################################################################################################################
    ################################################################## Newest version ##########################################################
    ############################################################################################################################################
    def __train(self, episode):
        # old actions mu (test for both the same as current actions and the previous policy)
        for traj in self.trajectories:
            traj['old_action_mus'] = self.generator([traj['states'], traj['codes']], training=False)
            # traj['old_actions'] = self.old_generator([traj['states'], traj['codes']], training=False) # use old generator weights
            # traj['old_actions'] = self.generator([traj['states'], traj['codes']], training=False)

        generated_states = np.concatenate([traj['states'] for traj in self.trajectories])
        generated_actions = np.concatenate([traj['actions'] for traj in self.trajectories])
        generated_codes = np.concatenate([traj['codes'] for traj in self.trajectories])
        generated_oldactions = np.concatenate([traj['old_action_mus'] for traj in self.trajectories])
        # generated_oldactions = np.concatenate([traj['old_actions'] for traj in self.trajectories])

        # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
        expert_idx = np.arange(self.expert_states.shape[0])
        np.random.shuffle(expert_idx)
        sampled_expert_states = self.expert_states[expert_idx, :]
        sampled_expert_actions = self.expert_actions[expert_idx, :]

        # sampled_generated_states = []
        # sampled_generated_actions = []
        # generated_idx = []
        # if generated_states.shape[0] > self.expert_states.shape[0]:
        #     generated_idx = np.random.choice(generated_states.shape[0], self.expert_states.shape[0], replace=False)
        # else:
        #     generated_idx = np.arange(generated_states.shape[0])
        #     np.random.shuffle(generated_idx)

        generated_idx = np.arange(generated_states.shape[0])
        np.random.shuffle(generated_idx)

        sampled_generated_states = generated_states[generated_idx, :]
        sampled_generated_actions = generated_actions[generated_idx, :]
        sampled_generated_states = tf.convert_to_tensor(sampled_generated_states, dtype=tf.float32)
        sampled_generated_actions = tf.convert_to_tensor(sampled_generated_actions, dtype=tf.float32)
        sampled_expert_states = tf.convert_to_tensor(sampled_expert_states, dtype=tf.float32)
        sampled_expert_actions = tf.convert_to_tensor(sampled_expert_actions, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((sampled_generated_states, sampled_generated_actions, sampled_expert_states, sampled_expert_actions))
        dataset = dataset.batch(batch_size=self.batch)

        # train discriminator
        loss = 0.0
        total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (generated_states_batch, generated_actions_batch, expert_states_batch, expert_actions_batch) in enumerate(dataset):
            with tf.GradientTape() as disc_tape:
                score1 = self.discriminator([generated_states_batch, generated_actions_batch], training=True)
                score2 = self.discriminator([expert_states_batch, expert_actions_batch], training=True)

                # wasserstein loss: D(x) - D(G(z))
                # score1 = tf.reduce_mean(score1)
                # score2 = -score2
                # score2 = tf.reduce_mean(score2)
                # disc_loss = tf.math.add(score1, score2)

                # cross entropy loss: D(G(z)) + (1 - D(x))
                disc_loss = self.__disc_loss(score1, score2)
            
            if save_loss: loss += tf.get_static_value(disc_loss) * generated_states_batch.shape[0]
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_weights))

            # for l in self.discriminator.layers:
            #     weights = l.get_weights()
            #     weights = [np.clip(w, -0.01, 0.01) for w in weights]
            #     l.set_weights(weights)
        
        if save_loss:
            episode_loss = loss / total_train_size
            episode_loss = episode_loss.item()
            self.disc_result.append(episode_loss)

        # train posterior
        generated_idx = np.arange(generated_states.shape[0])
        np.random.shuffle(generated_idx)
        sampled_generated_states = generated_states[generated_idx, :]
        sampled_generated_actions = generated_actions[generated_idx, :]
        sampled_generated_codes = generated_codes[generated_idx, :]
        sampled_generated_states = tf.convert_to_tensor(sampled_generated_states, dtype=tf.float32)
        sampled_generated_actions = tf.convert_to_tensor(sampled_generated_actions, dtype=tf.float32)
        sampled_generated_codes = tf.convert_to_tensor(sampled_generated_codes, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((sampled_generated_states, sampled_generated_actions, sampled_generated_codes))
        dataset = dataset.batch(batch_size=self.batch)

        loss = 0.0
        total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, actions_batch, codes_batch) in enumerate(dataset):
            with tf.GradientTape() as post_tape:
                prob = self.posterior([states_batch, actions_batch], training=True)

                post_loss = self.__post_loss(prob, codes_batch)
            
            if save_loss: loss += tf.get_static_value(post_loss) * states_batch.shape[0]
            gradients_of_posterior = post_tape.gradient(post_loss, self.posterior.trainable_weights)
            self.posterior_optimizer.apply_gradients(zip(gradients_of_posterior, self.posterior.trainable_weights))

        if save_loss:
            episode_loss = loss / total_train_size
            episode_loss = episode_loss.item()
            self.post_result.append(episode_loss)

        # TRPO
        # calculate rewards from discriminator and posterior
        for traj in self.trajectories:
            reward_d = (-tf.math.log(tf.keras.activations.sigmoid(self.discriminator([traj['states'], traj['actions']], training=False)))).numpy()
            reward_p = tf.math.log(self.posterior([traj['states'], traj['actions']], training=False)).numpy()

            traj['rewards'] = reward_d.flatten() + np.sum(reward_p * traj['codes'], axis=1)

            # calculate values, advants and returns
            values = self.value_net([traj['states'], traj['codes']], training=False).numpy().flatten() # Value function
            # baselines = np.append(values, values[-1])
            values_next = shift(values, -1, cval=0)
            deltas = traj['rewards'] + self.gamma * values_next - values # Advantage(st,at) = rt+1 + γ*V(st+1) - V(st)
            traj['advants'] = discount(deltas, self.gamma * self.lam)
            traj['returns'] = discount(traj['rewards'], self.gamma)

        advants = np.concatenate([traj['advants'] for traj in self.trajectories])

        advants /= (advants.std() + 1e-8)
        verybig = tf.where(tf.math.greater(np.absolute(advants), 50.0)).numpy().flatten()
        assert verybig.shape[0] == 0, "Very big!!!"

        # train value net for next iter
        returns = np.concatenate([traj['returns'] for traj in self.trajectories])

        generated_idx = np.arange(generated_states.shape[0])
        np.random.shuffle(generated_idx)
        sampled_generated_states = generated_states[generated_idx, :]
        sampled_generated_codes = generated_codes[generated_idx, :]
        sampled_returns = returns[generated_idx]
        sampled_generated_states = tf.convert_to_tensor(sampled_generated_states, dtype=tf.float32)
        sampled_generated_codes = tf.convert_to_tensor(sampled_generated_codes, dtype=tf.float32)
        sampled_returns = tf.convert_to_tensor(sampled_returns, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((sampled_generated_states, sampled_generated_codes, sampled_returns))
        dataset = dataset.batch(batch_size=self.batch)

        loss = 0.0
        total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, codes_batch, returns_batch) in enumerate(dataset):
            with tf.GradientTape() as value_tape:
                value_pred = self.value_net([states_batch, codes_batch], training=True)

                value_loss = self.__value_loss(value_pred, returns_batch)
            
            if save_loss: loss += tf.get_static_value(value_loss) * states_batch.shape[0]
            value_grads = value_tape.gradient(value_loss, self.value_net.trainable_weights)
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_weights))
        
        if save_loss:
            episode_loss = loss / total_train_size
            episode_loss = episode_loss.item()
            self.value_result.append(episode_loss)
        
        # save old policy (generator) before updating weights
        # old_weights = self.generator.get_weights()
        # self.old_generator.set_weights(old_weights)

        # generator training
        generated_idx = np.arange(generated_states.shape[0])
        np.random.shuffle(generated_idx)
        sampled_generated_states = generated_states[generated_idx, :]
        sampled_generated_codes = generated_codes[generated_idx, :]
        sampled_generated_actions = generated_actions[generated_idx, :]
        sampled_generated_oldactions = generated_oldactions[generated_idx, :]
        sampled_advants = advants[generated_idx]
        sampled_generated_states = tf.convert_to_tensor(sampled_generated_states, dtype=tf.float32)
        sampled_generated_codes = tf.convert_to_tensor(sampled_generated_codes, dtype=tf.float32)
        sampled_generated_actions = tf.convert_to_tensor(sampled_generated_actions, dtype=tf.float32)
        sampled_generated_oldactions = tf.convert_to_tensor(sampled_generated_oldactions, dtype=tf.float32)
        sampled_advants = tf.convert_to_tensor(sampled_advants, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((sampled_generated_states, sampled_generated_actions, sampled_generated_oldactions, sampled_generated_codes, sampled_advants))
        dataset = dataset.batch(batch_size=self.batch)

        loss = 0.0
        total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, actions_batch, oldactions_batch, codes_batch, advants_batch) in enumerate(dataset):
            # calculate previous theta (θold)
            thprev = get_flat(self.generator)
            # oldactions_batch = self.old_generator([states_batch, codes_batch], training=False)

            feed = {
                'states': states_batch,
                'actions': actions_batch,
                'codes': codes_batch,
                'advants': advants_batch,
                'old_actions': oldactions_batch
            }

            (surrogate_loss, grad_tape) = self.__generator_loss(feed)
            if save_loss:
                loss += tf.get_static_value(surrogate_loss) * states_batch.shape[0]

            policy_gradient = flatgrad(self.generator, surrogate_loss, grad_tape)
            nans = tf.math.is_nan(policy_gradient)
            if(tf.where(nans).numpy().flatten().shape[0] != 0): print('NAN!!!!!!!!!!!')
            stepdir = conjugate_gradient(self.fisher_vector_product, feed, policy_gradient.numpy()) # nan values here (inside fisher vector func)
            shs = stepdir.dot(self.fisher_vector_product(stepdir, feed)) # ...and here
            assert shs > 0

            lm = np.sqrt(shs / self.max_kl)
            fullstep = stepdir / lm
            neggdotstepdir = policy_gradient.numpy().dot(stepdir)

            # save old policy (generator) before updating weights
            # old_weights = self.generator.get_weights()
            # self.old_generator.set_weights(old_weights)

            theta = linesearch(self.get_loss, thprev, feed, fullstep, neggdotstepdir / lm)
            # set_from_flat(self.generator, theta)
            var_list = self.generator.trainable_weights
            shapes = [v.shape for v in var_list]
            start = 0

            weight_idx = 0
            for shape in shapes:
                size = np.prod(shape)
                self.generator.trainable_weights[weight_idx].assign(tf.reshape(theta[start:start + size], shape))
                weight_idx += 1
                start += size

        if save_loss:
            episode_loss = loss / total_train_size
            episode_loss = episode_loss.item()
            self.gen_result.append(episode_loss)
            if episode % 2 == 0 and episode != 0: self.show_loss()

        if save_models:
            self.generator.save_weights('./saved_models/trpo/generator.h5')
            self.discriminator.save_weights('./saved_models/trpo/discriminator.h5')
            self.posterior.save_weights('./saved_models/trpo/posterior.h5')
            self.value_net.save_weights('./saved_models/trpo/value_net.h5')
            # self.old_generator.save_weights('./saved_models/trpo/old_generator.h5')
            yaml_conf = {
                'episode': episode+1,
                'gen_loss': self.gen_result,
                'disc_loss': self.disc_result,
                'post_loss': self.post_result,
                'value_loss': self.value_result
            }
            
            with open("./saved_models/trpo/model.yml", 'w') as f:
                yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)
    
    def infogail(self):
        # load data
        expert_states, expert_actions, expert_codes = pkl.load(open("expert_traj.pkl", "rb"))

        self.expert_states = np.concatenate(expert_states)
        self.expert_actions = np.concatenate(expert_actions)
        self.expert_codes = np.concatenate(expert_codes)

        for episode in trange(self.starting_episode, self.episodes, desc="Episode"):
            # Sample a batch of latent codes: ci ∼ p(c)
            sampled_codes = np.zeros((self.code_batch, self.code_dims))
            for i in range(self.code_batch):
                pick = np.random.choice(self.code_dims, 1)[0]
                sampled_codes[i, pick] = 1
            
            # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
            self.trajectories = []

            for i in range(len(sampled_codes)):
                trajectory_dict = {}
                trajectory = self.__generate_policy(sampled_codes[i])
                trajectory_dict['states'] = np.copy(trajectory[0])
                trajectory_dict['actions'] = np.copy(trajectory[1])
                trajectory_dict['codes'] = np.copy(trajectory[2])
                self.trajectories.append(trajectory_dict)

            # call train here
            self.__train(episode)
        
        # save useful stuff
        if save_loss:
            self.show_loss()

# main
agent = CircleAgent(10, 2, 3)
agent.infogail()