"""Hyper parameters for the ARS algorithm"""

NUM_STEPS = 700    # number of iterations to train for
MAX_EPISODE_LENGTH = 1000   # maximum length of an episode
NUM_DELTAS = 16 # number of deltas to evaluate
NUM_BEST_DELTAS = 16    # number of best deltas
assert NUM_BEST_DELTAS <= NUM_DELTAS    # NUM_BEST_DELTAS should always be <= NUM_DELTAS

ALPHA = 0.02    # learning rate (step-size)
NOISE = 0.03    # strength of the deltas
