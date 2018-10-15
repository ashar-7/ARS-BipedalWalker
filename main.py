import pickle
import gym
import numpy as np

from ARS.agent import ARS_Agent
from ARS.normalizer import Normalizer

ENV_NAME = 'BipedalWalker-v2'     # Environment name
MONITOR_DIR = 'ARS_Agent_Monitor' # Directory to save the monitored environment data in
LOG_FILENAME = 'Rewards_Log.txt'  # File to log the rewards to

# create the environment
env = gym.make(ENV_NAME)

state_len = env.observation_space.shape[0]  # size (or length) of a state
action_len = env.action_space.shape[0]      # size (or length) of an action

# state normalizer
normalizer = Normalizer(state_len)

# create the agent
agent = ARS_Agent(state_len, action_len)

# wrap the environment with the monitor wrapper
env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=agent.should_record, force=False)

# train the agent
record_every = 10   # we'll record the video after every record_every steps
log_every = 10      # we'll log the rewards after every log_every steps
agent.train(env, normalizer, LOG_FILENAME, log_every, record_every)

# save the normalizer object
with open('ARS/normalizer.pickle', 'wb') as f:
    pickle.dump(normalizer, f)

# save the agent's weights
np.save('ARS/agent_weights.npy', agent.weights)
