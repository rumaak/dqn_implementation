import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.transform import resize

from dqn_implementation.model import DQN
from dqn_implementation.data.utilities import rgb2grayscale


class Predictor:
    """ Model wrapper enabling training and prediction """
    def __init__(self, replay_memory_size, stacked_frames, env, action_count,
                 replay_start_size, action_repeat, update_frequency, noop_max,
                 scaling_factor, network_update_frequency,
                 models_directory, device):

        # These three together form the memory consisting of experiences
        # e = (s,a,r,s')
        self.states = np.zeros(shape=(replay_memory_size, stacked_frames, 84,
                                      84))
        self.actions = np.zeros(shape=(replay_memory_size))
        self.rewards = np.zeros(shape=(replay_memory_size))

        self.total_frames = 0
        self.memory_frames = 0
        self.episodes_end = np.zeros((replay_memory_size), dtype=bool)

        self.env = env
        self.stacked_frames = stacked_frames
        self.replay_start_size = replay_start_size
        self.replay_memory_size = replay_memory_size
        self.action_repeat = action_repeat
        self.update_frequency = update_frequency
        self.noop_max = noop_max
        self.scaling_factor = scaling_factor
        self.network_update_frequency = network_update_frequency
        self.models_directory = models_directory
        self.action_count = action_count
        self.device = device

        self.dqn_current = DQN(action_count).to(device)
        self.dqn_new = DQN(action_count).to(device)
        self.dqn_new.load_state_dict(self.dqn_current.state_dict())

        self.losses = None
        self.reward_sums = None

    def set_model(self, model):
        self.dqn_current = model.to(self.device)
        self.dqn_new = DQN(self.action_count).to(self.device)
        self.dqn_new.load_state_dict(self.dqn_current.state_dict())

    def set_model_from_file(self, filepath):
        model = DQN(self.action_count).to(self.device)
        model.load_state_dict(torch.load(filepath))
        self.set_model(model)

    def train(self, epochs, optimizer, batch_size, stages, stage,
              end_eps_frame, eps_beg, eps_end, cp_name="dqn_checkpoint"):

        eps_step = (eps_beg - eps_end) / end_eps_frame

        self.dqn_current.eval()
        self.dqn_new.train()

        if (self.losses is None) or (self.reward_sums is None):
            self.losses = np.zeros((stages, epochs))
            self.reward_sums = np.zeros((stages, epochs))

        for epoch in range(epochs):
            losses = 0

            self.env.reset()

            last_frames = np.zeros(shape=(self.stacked_frames, 84, 84))
            last_frame = np.zeros(shape=(210, 160, 3))

            previous_action = None

            noops = 0
            episode_frames = 0

            while(True):
                # Interact with environment
                epsilon = max(eps_beg - (self.total_frames *
                                         eps_step), eps_end)

                action = self.__choose_action(episode_frames, epsilon,
                                              last_frames, previous_action)
                observation, reward, done, info = self.env.step(action)

                # Clip reward
                reward = min(reward, 1.0)

                # Saving data for memory replay
                if episode_frames > (self.stacked_frames - 1):
                    self.states[self.memory_frames % self.replay_memory_size] = last_frames
                    self.actions[self.memory_frames % self.replay_memory_size] = action
                    self.rewards[self.memory_frames % self.replay_memory_size] = reward

                    # If this is the last point of episode, mark it in episodes_end array
                    if done or (noops >= 30):
                        self.episodes_end[self.memory_frames % self.replay_memory_size] = True
                    else:
                        self.episodes_end[self.memory_frames % self.replay_memory_size] = False

                    self.memory_frames += 1

                previous_action = action

                # Max over last two frames, conversion RGB -> grayscale, rescale
                frame = np.where(last_frame > observation, last_frame, observation)
                frame = rgb2grayscale(frame)
                frame = resize(frame, (84, 84))
                frame = frame / 255.0

                # Stacking last four frames and assigning last frame (observation)
                last_frames = np.concatenate((last_frames[1:], frame[None, :]))
                last_frame = observation

                # Training
                if self.memory_frames >= self.replay_start_size:
                    if self.total_frames % (self.update_frequency * self.action_repeat) == 0:
                        # -1 because of shifted states
                        experience_count = min(self.memory_frames, self.replay_memory_size - 1)
                        episodes_end_indices = np.array(self.episodes_end).nonzero()[0]
                        batch_indices = np.random.choice(range(experience_count-1), batch_size, False)

                        # Sample states, actions, rewards and new states
                        s = torch.from_numpy(self.states[batch_indices]).float().to(self.device)
                        a = torch.from_numpy(self.actions[batch_indices]).float().to(self.device)
                        r = torch.from_numpy(self.rewards[batch_indices]).float().to(self.device)
                        sn = torch.from_numpy(self.states[batch_indices+1]).float().to(self.device)

                        # Mask out rewards of new states when new state isn't terminal
                        episodes_end_mask = torch.from_numpy(
                            (batch_indices[:, None] == episodes_end_indices[None, :]).any(axis=1)).to(self.device)

                        # Select only outputs with maximal reward
                        output_current = torch.max(self.dqn_current(sn), 1).values

                        # Select actions that were originally selected
                        output_new = self.dqn_new(s)[range(batch_size),
                                                     a.to(torch.long)]

                        # IF current state is the last state of episode,
                        # don't compute value of new state
                        output_current = torch.where(episodes_end_mask,
                                                     torch.zeros(
                                                         batch_size,
                                                         device=self.device),
                                                     output_current).detach()

                        target = r + (self.scaling_factor * output_current)

                        optimizer.zero_grad()

                        loss = F.smooth_l1_loss(output_new, target)
                        loss.backward()

                        for param in self.dqn_new.parameters():
                                param.grad.data.clamp_(-1, 1)

                        optimizer.step()

                        self.losses[stage, epoch] += loss.detach()
                        losses += 1


                    if self.total_frames % self.network_update_frequency == 0:
                        self.dqn_current.load_state_dict(
                            self.dqn_new.state_dict())

                self.reward_sums[stage, epoch] += reward

                if action == 0 and episode_frames < self.noop_max:
                    noops += 1
                else:
                    noops = 0

                episode_frames += 1
                self.total_frames += 1

                if done or (noops >= 30):
                    noops = 0
                    break

            if losses != 0:
                self.losses[stage, epoch] = self.losses[stage, epoch] / losses

        torch.save(self.dqn_new.state_dict(),f'{self.models_directory}{cp_name}.pt')

        self.env.close()

    def __plot(self, data, stages, epochs, title):
        if stages is not None:
            data = data[stages]
            if epochs is not None:
                assert (not isinstance(stages, list)), \
                       "Can select values from one stage only"
                data = data[epochs]

        data = np.reshape(data, -1)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(data)+1), data)
        if title is not None:
            ax.set_title(title)

        ax.grid()

    def plot_losses(self, stages=None, epochs=None, title=None):
        """ Can be used after full training """
        if self.losses is None:
            print("Nothing to plot")
            return

        self.__plot(self.losses, stages, epochs, title)

    def plot_rewards(self, stages=None, epochs=None, title=None):
        """ Can be used after full training """
        if self.reward_sums is None:
            print("Nothing to plot")
            return

        self.__plot(self.reward_sums, stages, epochs, title)

    def visualize(self, epochs, epsilon=0.05, render=False, frame_delay=0.01,
                 batch_size=32, time_limit=None):

        self.dqn_current.eval()
        self.dqn_new.eval()

        rewards = []
        t0 = time.time()

        epoch = 0
        while epoch < epochs:
            self.env.reset()

            if time_limit is not None:
                t1 = time.time()
                if t1 - t0 > time_limit:
                    break

            last_frames = np.zeros(shape=(self.stacked_frames, 84, 84))
            last_frame = np.zeros(shape=(210, 160, 3))

            previous_action = None

            rewards.append(0)
            noops = 0
            episode_frames = 0

            while(True):

                if render:
                    self.env.render()
                    time.sleep(frame_delay)

                action = self.__choose_action(episode_frames, epsilon,
                                              last_frames, previous_action)
                observation, reward, done, info = self.env.step(action)

                # Clip reward
                reward = min(reward, 1.0)

                previous_action = action

                # Max over last two frames, conversion RGB -> grayscale, rescale
                frame = np.where(last_frame > observation, last_frame, observation)
                frame = rgb2grayscale(frame)
                frame = resize(frame, (84, 84))
                frame = frame / 255.0

                # Stacking last four frames and assigning last frame (observation)
                last_frames = np.concatenate((last_frames[1:], frame[None, :]))
                last_frame = observation

                if action == 0 and episode_frames < self.noop_max:
                    noops += 1
                else:
                    noops = 0

                episode_frames += 1
                rewards[-1] += reward

                if done or (noops >= 30):
                    noops = 0
                    break

            if time_limit is None:
                epoch += 1

        self.env.close()
        return rewards

    def __choose_action(self, episode_frames, epsilon, last_frames,
                        previous_action):
        action = None
        if episode_frames % self.action_repeat == 0:
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                inp_torch = torch.from_numpy(last_frames[None, :]).float().to(self.device)
                action_torch = self.dqn_current(inp_torch).argmax(axis=1)[0]
                # if episode_frames > 50:
                #     print(self.dqn_current(inp_torch))
                action = action_torch.cpu().detach().numpy()
        else:
            action = previous_action

        return action
