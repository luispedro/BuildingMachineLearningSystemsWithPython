#!/usr/bin/env python

# Built by merging different Q examples available online

import gym

import os
import numpy as np
import tensorflow as tf
import random
from collections import deque

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))[None,:,:]

def adapt_state(state):
    return [np.float32(np.transpose(state, (2, 1, 0)) / 255.0)]

env_name = "Breakout-v4"

width = 80  # Resized frame width
height = 105  # Resized frame height

n_episodes = 12000  # Number of runs for the agent
state_length = 4  # Number of most frames we input to the network

gamma = 0.99  # Discount factor

exploration_steps = 1000000  # During all these steps, we progressively lower epsilon
initial_epsilon = 1.0  # Initial value of epsilon in epsilon-greedy
final_epsilon = 0.1  # Final value of epsilon in epsilon-greedy

initial_random_search = 20000  # Number of steps to populate the replay memory before training starts
replay_memory_size = 400000  # Number of states we keep for training
batch_size = 32  # Batch size
network_update_interval = 10000  # The frequency with which the target network is updated
train_skips = 4  # The agent selects 4 actions between successive updates

learning_rate = 0.00025  # Learning rate used by RMSProp
momentum = 0.95  # momentum used by RMSProp
min_gradient = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

network_path = 'saved_networks/' + env_name
tensorboard_path = 'summary/' + env_name
save_interval = 300000  # The frequency with which the network is saved
initial_quiet_steps = 30  # Initial steps while the agent is not doing anything. We keep this to start populating the state


class Agent():
    def __init__(self, num_actions, restore_network=False):
        self.num_actions = num_actions
        self.epsilon = initial_epsilon
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(tensorboard_path, self.sess.graph)

        if not os.path.exists(network_path):
            os.makedirs(network_path)
            
        # Initialize target network
        self.sess.run(self.update_target_network)

        if restore_network:
            self.load_network()
            
    def build_network(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu', input_shape=(width, height, state_length), name="Layer1"))
        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', name="Layer2"))
        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name="Layer3"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', name="Layer4"))
        model.add(tf.keras.layers.Dense(self.num_actions, name="Output"))

        s = tf.placeholder(tf.float32, [None, width, height, state_length])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.matmul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, epsilon=min_gradient)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, frame):
        processed_frame = preprocess(frame)
        state = [processed_frame for _ in range(state_length)]
        return np.concatenate(state)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < initial_random_search:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: adapt_state(state)}))

        # Anneal epsilon linearly over time
        if self.epsilon > final_epsilon and self.t >= initial_random_search:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, frame):
        next_state = np.append(state[1:, :, :], frame, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > replay_memory_size:
            self.replay_memory.popleft()

        if self.t >= initial_random_search:
            # Train network
            if self.t % train_skips == 0:
                self.train_network()

            # Update target network
            if self.t % network_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % save_interval == 0:
                save_path = self.saver.save(self.sess, network_path + '/' + env_name, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: adapt_state(state)}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= initial_random_search:
                stats = [self.total_reward, self.total_q_max / self.duration,
                        self.duration, self.total_loss / (self.duration / train_skips)]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < initial_random_search:
                mode = 'random'
            elif initial_random_search <= self.t < initial_random_search + exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(train_skips)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * gamma * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        action = np.argmax(self.q_values.eval(feed_dict={self.s: adapt_state(state)}))
        return action

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = Agent(num_actions=env.action_space.n)

    for _ in range(n_episodes):
        terminal = False
        frame = env.reset()
        for _ in range(random.randint(1, initial_quiet_steps)):
            frame, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(frame)
        while not terminal:
            action = agent.get_action(state)
            frame, reward, terminal, _ = env.step(action)

            processed_frame = preprocess(frame)
            state = agent.run(state, action, reward, terminal, processed_frame)

    frame = env.reset()
    env.render()
    
    frame, _, is_done, _ = env.step(0)  # Do nothing
    state = agent.get_initial_state(frame)

    while not is_done:
        frame, reward, is_done, _ = env.step(agent.get_action_at_test(state))
        env.render()
        processed_frame = preprocess(frame)
        next_state = np.append(state[1:, :, :], processed_frame, axis=0)
