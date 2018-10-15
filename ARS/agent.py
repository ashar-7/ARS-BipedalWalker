"""ARS Agent"""

import numpy as np
import ARS.hyper_params as hp

class ARS_Agent():

    def __init__(self, num_inputs, num_outputs):
        # inputs are the states, outputs are the actions

        self.num_inputs = num_inputs    # length of a state
        self.num_outputs = num_outputs  # length of an action
        self.weights = np.zeros((num_outputs, num_inputs))

        self.record_video = False                           # flag for recording video
        self.should_record = lambda i: self.record_video    # video callable for recording video

    def generate_deltas(self, N):
        """Returns a list of N random deltas"""

        return [np.random.randn(*self.weights.shape) for _ in range(N)]

    def policy(self, state, delta=None, direction=None):
        """Given a state, returns an action to take in that state
        If delta and direction both are not None, returns an action by
        considering them along with the weights"""

        # Assertion:
        if direction:
            assert direction is '+' or direction is '-', "direction should either be '+' or '-'"

        if delta is not None:
            if direction == "+":
                return (self.weights + hp.NOISE * delta).dot(state)
            elif direction == "-":
                return (self.weights - hp.NOISE * delta).dot(state)
        else:
            return self.weights.dot(state)

        # return None if delta was given but no direction was given
        return None

    def run_episode(self, env, normalizer, delta=None, direction=None, render=False):
        """Generates an episode and returns the total reward recieved"""

        total_reward = 0
        state = env.reset()
        for _ in range(hp.MAX_EPISODE_LENGTH):
            # render if render == True
            if render:
                env.render()

            normalizer.observe(state) # observe running mean and variance
            state = normalizer.normalize(state) # normalize the state

            action = self.policy(state, delta=delta, direction=direction) # select an action
            state, reward, done, _ = env.step(action)

            reward = max(min(reward, 1), -1) # clip reward between -1 and 1
            total_reward += reward

            if done:
                break

        env.env.close()
        return total_reward

    def update_weights(self, rollouts, sigma_R):
        """Makes a weight update using rollouts and standard deviation of the rewards."""

        step = np.zeros(self.weights.shape)

        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta

        self.weights += hp.ALPHA / (hp.NUM_BEST_DELTAS * sigma_R) * step

    def set_weights(self, weights):
        """Sets self.weights = weights while ensuring the size of both are same"""

        assert weights.shape == self.weights.shape, "weights.shape should be equal to self.weights.shape"

        self.weights = weights

    def train(self, env, normalizer, log_filename, log_every, record_every):
        """Trains the agent with the ARS algorithm"""

        # If env.spec.timestep_limit is available, set the max episode length to it
        # otherwise it would cause some recording issues
        hp.MAX_EPISODE_LENGTH = env.spec.timestep_limit or hp.MAX_EPISODE_LENGTH

        # loop for the number of steps to train for
        for step in range(hp.NUM_STEPS):
            # generate deltas
            deltas = self.generate_deltas(hp.NUM_DELTAS)

            # explore positive and negative deltas and record the rewards
            pos_delta_rewards = [0] * hp.NUM_DELTAS
            neg_delta_rewards = [0] * hp.NUM_DELTAS

            for i in range(hp.NUM_DELTAS):
                pos_delta_rewards[i] = self.run_episode(env, normalizer, deltas[i], '+')
                neg_delta_rewards[i] = self.run_episode(env, normalizer, deltas[i], '-')

            # concat both delta rewards together and convert it to a numpy array,
            # then calculate their standard deviation
            all_delta_rewards = np.array(pos_delta_rewards + neg_delta_rewards)
            sigma_R = all_delta_rewards.std()

            # sort the rollouts by pos_delta_rewards and neg_delta_rewards
            rollouts = [(pos_delta_rewards[i], neg_delta_rewards[i], deltas[i]) for i in range(hp.NUM_DELTAS)]
            rollouts.sort(key=lambda x: max(x[0:2]),reverse=True)
            rollouts = rollouts[:hp.NUM_BEST_DELTAS] # select best rollouts

            # make the weights update
            self.update_weights(rollouts, sigma_R)

            if step % record_every == 0:
                self.record_video = True

            # evaluate current weights
            reward = self.run_episode(env, normalizer)
            self.record_video = False

            # log the rewards every log_every steps
            if step % log_every == 0:
                with open(log_filename, 'a') as f:
                    f.write("Step: #{} Reward: {}\n".format(step, reward))

            print("Step: #{} Reward: {}".format(step, reward))
