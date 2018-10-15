import pickle
import gym
import numpy as np

from ARS.agent import ARS_Agent
from ARS.normalizer import Normalizer

ENV_NAME = 'BipedalWalker-v2'     # Environment name
MONITOR_DIR = 'ARS_Agent_Monitor' # Directory to save the monitored environment data in
LOG_FILENAME = 'Rewards_Log.txt'  # File to log the rewards to
NORMALIZER_FILENAME = 'normalizer.pickle' # Filename to load the normalizer object from
WEIGHTS_FILENAME = 'agent_weights.npy' # Filename to load the rewards from

# create the environment
env = gym.make(ENV_NAME)

state_len = env.observation_space.shape[0]  # size (or length) of a state
action_len = env.action_space.shape[0]      # size (or length) of an action

# state normalizer
with open(NORMALIZER_FILENAME, 'rb') as f:
    normalizer = pickle.load(f)

normalizer = Normalizer(state_len)

# create the agent
agent = ARS_Agent(state_len, action_len)

# wrap the environment with the monitor wrapper
env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=agent.should_record, force=False)

# load and set the weights
weights = np.load(WEIGHTS_FILENAME)
agent.set_weights(weights)

# run a test episode
agent.run_episode(env, normalizer, render=True)
