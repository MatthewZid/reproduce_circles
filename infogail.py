import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import yaml
import time

from circle_env import CircleEnv
from tqdm import trange
from trpo import *
import trpo
from scipy.ndimage import shift
import multiprocessing as mp
from models import *

# tf.config.threading.set_inter_op_parallelism_threads(16) 
# tf.config.threading.set_intra_op_parallelism_threads(16)
# tf.config.set_soft_device_placement(True)

class Agent():
    def __generate_trajectory(self, code):
        s_traj = []
        a_traj = []
        c_traj = []
        env = CircleEnv()

        # generate actions for every current state
        state_obsrv = env.reset() # reset environment state
        code_tf = tf.constant(code)
        code_tf = tf.expand_dims(code_tf, axis=0)

        while True:
            # 1. generate actions with generator
            state_tf = tf.constant(state_obsrv)
            state_tf = tf.expand_dims(state_tf, axis=0)
            action_mu = models.generator.model([state_tf, code_tf], training=False)
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
    
    def run(self, code):
        try:
            trajectory_dict = {}
            trajectory = self.__generate_trajectory(code)
            trajectory_dict['states'] = np.copy(trajectory[0])
            trajectory_dict['actions'] = np.copy(trajectory[1])
            trajectory_dict['codes'] = np.copy(trajectory[2])
            return trajectory_dict
        except KeyboardInterrupt:
            time.sleep(1)

class InfoGAIL():
    def __init__(self, batch_size=2048, code_batch=384, episodes=10000, gamma=0.997, lam=0.97):
        self.batch = batch_size
        self.code_batch = code_batch
        self.episodes = episodes
        self.gamma = gamma
        self.lam = lam
        self.starting_episode = 0
        self.gen_result = []
        self.disc_result = []
        self.post_result = []
        self.value_result = []
        self.total_rewards = []

        # load data
        expert_states, expert_actions, expert_codes = pkl.load(open("expert_traj.pkl", "rb"))

        self.expert_states = np.concatenate(expert_states)
        self.expert_actions = np.concatenate(expert_actions)
        self.expert_codes = np.concatenate(expert_codes)

        # create replay buffer
        self.buffer = ReplayBuffer(int(code_batch / BUFFER_RATIO), code_batch)

        generator_weight_path = ''
        if resume_training:
            with open("./saved_models/trpo/model.yml", 'r') as f:
                data = yaml.safe_load(f)
                self.starting_episode = data['episode']
                print('\nRestarting from episode {:d}'.format(self.starting_episode))
                self.gen_result = data['gen_loss']
                self.disc_result = data['disc_loss']
                self.post_result = data['post_loss']
                self.value_result = data['value_loss']
            generator_weight_path = './saved_models/trpo/generator.h5'
            models.discriminator.model.load_weights('./saved_models/trpo/discriminator.h5')
            models.posterior.model.load_weights('./saved_models/trpo/posterior.h5')
            models.value_net.model.load_weights('./saved_models/trpo/value_net.h5')
        else:
            generator_weight_path = './saved_models/bc/generator.h5'
        
        models.generator.model.load_weights(generator_weight_path)
        print('\nSetup ready!')
    
    def __saveplot(self, x, y, episode, element='element', mode='plot'):
        plt.figure()
        if mode == 'plot': plt.plot(x, y)
        else: plt.scatter(x, y, alpha=0.4)
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
        # plt.ylim(-100, 100)
        plt.title('Surrogate loss')
        plt.plot(epoch_space, self.gen_result)
        plt.savefig('./plots/trpo_loss', dpi=100)
        plt.close()

        plt.figure()
        plt.title('Disc/Post losses')
        plt.plot(epoch_space, self.disc_result)
        plt.plot(epoch_space, self.post_result)
        plt.legend(['disc', 'post'], loc="lower left")
        plt.savefig('./plots/disc_post', dpi=100)
        plt.close()

        plt.figure()
        plt.title('Value loss')
        plt.plot(epoch_space, self.value_result)
        plt.savefig('./plots/value_loss', dpi=100)
        plt.close()
    
    def train(self, agent):
        for episode in trange(self.starting_episode, self.episodes, desc="Episode"):
            # Sample a batch of latent codes: ci ∼ p(c)
            sampled_codes = np.zeros((self.code_batch, models.generator.code_dims))
            for i in range(self.code_batch):
                pick = np.random.choice(models.generator.code_dims, 1)[0]
                sampled_codes[i, pick] = 1
            
            # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
            trajectories = []
            with mp.Pool(mp.cpu_count()) as pool:
                trajectories = pool.map(agent.run, sampled_codes)
            
            # Sample from buffer
            # for traj in trajectories:
            #     self.buffer.add(traj)
            # trajectories = self.buffer.sample()
            
            for traj in trajectories:
                traj['old_action_mus'] = normalize_mu(models.generator.model([traj['states'], traj['codes']], training=False))
            
            generated_states = np.concatenate([traj['states'] for traj in trajectories])
            generated_actions = np.concatenate([traj['actions'] for traj in trajectories])
            generated_codes = np.concatenate([traj['codes'] for traj in trajectories])
            generated_oldactions = np.concatenate([traj['old_action_mus'] for traj in trajectories])

            # train discriminator
            # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
            expert_idx = np.arange(self.expert_states.shape[0])
            np.random.shuffle(expert_idx)
            # expert_idx = np.random.choice(self.expert_states.shape[0], int(self.expert_states.shape[0] / 2), replace=False)
            shuffled_expert_states = self.expert_states[expert_idx, :]
            shuffled_expert_actions = self.expert_actions[expert_idx, :]

            generated_idx = np.arange(generated_states.shape[0])
            np.random.shuffle(generated_idx)
            # generated_idx = np.random.choice(generated_states.shape[0], int(generated_states.shape[0] / 2), replace=False)
            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_actions = generated_actions[generated_idx, :]

            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float32)
            shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float32)
            shuffled_expert_states = tf.convert_to_tensor(shuffled_expert_states, dtype=tf.float32)
            shuffled_expert_actions = tf.convert_to_tensor(shuffled_expert_actions, dtype=tf.float32)

            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_expert_states, shuffled_expert_actions))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.discriminator.train(dataset)
            if save_loss: self.disc_result.append(loss)

            # train posterior
            generated_idx = np.arange(generated_states.shape[0])
            np.random.shuffle(generated_idx)
            # shuffled_generated_states = np.concatenate([self.expert_states, generated_states], axis=0)
            # shuffled_generated_actions = np.concatenate([self.expert_actions, generated_actions], axis=0)
            # shuffled_generated_codes = np.concatenate([self.expert_codes, generated_codes], axis=0)
            # generated_idx = np.arange(shuffled_generated_states.shape[0])
            # np.random.shuffle(generated_idx)
            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_actions = generated_actions[generated_idx, :]
            shuffled_generated_codes = generated_codes[generated_idx, :]
            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float32)
            shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float32)
            shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_generated_codes))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.posterior.train(dataset)
            if save_loss: self.post_result.append(loss)

            # TRPO/PPO
            # calculate rewards from discriminator and posterior
            episode_rewards = []
            for traj in trajectories:
                reward_d = (-tf.math.log(tf.keras.activations.sigmoid(models.discriminator.model([traj['states'], traj['actions']], training=False)))).numpy()
                reward_p = models.posterior.model([traj['states'], traj['actions']], training=False).numpy()
                reward_p = np.sum(np.ma.log(reward_p).filled(0) * traj['codes'], axis=1).flatten() # np.ma.log over tf.math.log, fixes log of zero

                traj['rewards'] = 0.6 * reward_d.flatten() + 0.4 * reward_p
                episode_rewards.append(traj['rewards'].sum())

                # calculate values, advants and returns
                values = models.value_net.model([traj['states'], traj['codes']], training=False).numpy().flatten() # Value function
                values_next = shift(values, -1, cval=0)
                deltas = traj['rewards'] + self.gamma * values_next - values # Advantage(st,at) = rt+1 + γ*V(st+1) - V(st)
                traj['advants'] = discount(deltas, self.gamma * self.lam)
                traj['returns'] = discount(traj['rewards'], self.gamma)
            
            advants = np.concatenate([traj['advants'] for traj in trajectories])
            # advants /= advants.std()

            # train value net for next iter
            returns = np.expand_dims(np.concatenate([traj['returns'] for traj in trajectories]), axis=1)

            generated_idx = np.arange(generated_states.shape[0])
            np.random.shuffle(generated_idx)
            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_codes = generated_codes[generated_idx, :]
            shuffled_returns = returns[generated_idx, :]
            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float32)
            shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float32)
            shuffled_returns = tf.convert_to_tensor(shuffled_returns, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_codes, shuffled_returns))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.value_net.train(dataset)
            if save_loss: self.value_result.append(loss)

            # generator training
            feed = {
                'states': generated_states,
                'actions': generated_actions,
                'codes': generated_codes,
                'advants': advants,
                'old_mus': generated_oldactions
            } 

            loss = models.generator.train(feed)
            if save_loss: self.gen_result.append(loss)

            if save_loss:
                # plot rewards and losses
                episode_reward = np.array(episode_rewards, dtype=np.float32).mean()
                self.total_rewards.append(episode_reward)
                if episode != 0:
                    self.show_loss()
                    epoch_space = np.arange(1, len(self.total_rewards)+1, dtype=int)
                    self.__saveplot(epoch_space, self.total_rewards, 0, 'rewards')
            
            if episode != 0 and (episode % 100 == 0): print('Theta updates so far: {:d}'.format(trpo.improved))

            if save_models:
                models.generator.model.save_weights('./saved_models/trpo/generator.h5')
                models.discriminator.model.save_weights('./saved_models/trpo/discriminator.h5')
                models.posterior.model.save_weights('./saved_models/trpo/posterior.h5')
                models.value_net.model.save_weights('./saved_models/trpo/value_net.h5')
                yaml_conf = {
                    'episode': episode+1,
                    'gen_loss': self.gen_result,
                    'disc_loss': self.disc_result,
                    'post_loss': self.post_result,
                    'value_loss': self.value_result
                }
                
                with open("./saved_models/trpo/model.yml", 'w') as f:
                    yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)

models = Models()

# main
def main():
    agent = Agent()
    infogail = InfoGAIL()
    infogail.train(agent)

if __name__ == '__main__':
    main()