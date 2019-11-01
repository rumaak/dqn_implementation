import gym
import numpy as np
from skimage.transform import resize

from .utilities import rgb2grayscale


def generate(amount):
    """
    Generate given amount of data from environment using random agent.
    """
    stacked_frames = 4
    memory_size = amount

    # These four together form the memory consisting of
    # experiences e = (s,a,r,s')
    states = np.zeros(shape=(memory_size, 84, 84, stacked_frames))
    actions = np.zeros(shape=(memory_size))
    rewards = np.zeros(shape=(memory_size))
    new_states = np.zeros(shape=(memory_size, 84, 84, stacked_frames))

    last_frames = np.zeros(shape=(stacked_frames, 84, 84))
    last_frame = np.zeros(shape=(210, 160, 3))

    previous_state = np.zeros(shape=(stacked_frames, 84, 84))
    previous_action = None
    previous_reward = None

    env = gym.make('Breakout-v0')
    observation = env.reset()
    for t in range(amount + stacked_frames):
        # Interact with environment
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # Max over last two frames, conversion RGB -> grayscale, rescale
        frame = np.where(last_frame > observation, last_frame, observation)
        frame = rgb2grayscale(frame)
        frame = resize(frame, (84, 84))

        # Stacking last four frames and assigning last frame (observation)
        last_frames = np.concatenate((last_frames[1:], frame[None, :]))
        last_frame = observation

        # Saving data for memory replay
        if t >= (stacked_frames):
            states[t % memory_size] = previous_state
            actions[t % memory_size] = previous_action
            rewards[t % memory_size] = previous_reward
            new_states[t % memory_size] = np.moveaxis(last_frames, 0, 2)

        # Keep to save in next iteration together with new state
        previous_state = np.moveaxis(last_frames, 0, 2)
        previous_action = action
        previous_reward = reward

        if done:
            env.reset()

    env.close()
    return states, actions, rewards, new_states
