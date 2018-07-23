#!/usr/bin/env python

# Built by merging different Q examples available online

import gym

import os
import six
import numpy as np
import tensorflow as tf
import random
import itertools
from collections import deque, namedtuple

CHART_DIR = "charts"
if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)

env_name = "Breakout-v4"

width = 80  # Resized frame width
height = 105  # Resized frame height

n_episodes = 12000  # Number of runs for the agent
state_length = 4  # Number of most frames we input to the network

gamma = 0.99  # Discount factor

exploration_steps = 1000000  # During all these steps, we progressively lower epsilon
initial_epsilon = 1.0  # Initial value of epsilon in epsilon-greedy
final_epsilon = 0.1  # Final value of epsilon in epsilon-greedy

replay_memory_init_size = 1000  # Number of steps to populate the replay memory before training starts
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

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))[None,:,:]

def adapt_state(state):
    return [np.float32(np.transpose(state, (2, 1, 0)) / 255.0)]

def adapt_batch_state(state):
    return np.transpose(np.array(state), (0, 3, 2, 1)) / 255.0

def get_initial_state(frame):
    processed_frame = preprocess(frame)
    state = [processed_frame for _ in range(state_length)]
    return np.concatenate(state)

class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, env, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.num_actions = env.action_space.n
        self.epsilon = initial_epsilon
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self.build_model()
        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_%s" % scope)
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_model(self):
        """
        Builds the Tensorflow graph.
        """
        self.X = tf.placeholder(shape=[None, width, height, state_length], dtype=tf.float32, name="X")
        # The TD target value
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        model = tf.keras.Sequential(name=self.scope)
        model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu', input_shape=(width, height, state_length), name="Layer1"))
        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', name="Layer2"))
        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name="Layer3"))
        model.add(tf.keras.layers.Flatten(name="Flatten"))
        model.add(tf.keras.layers.Dense(512, activation='relu', name="Layer4"))
        model.add(tf.keras.layers.Dense(self.num_actions, name="Output"))

        self.predictions = model(self.X)

        a_one_hot = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.predictions, a_one_hot), reduction_indices=1)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y, q_value)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, epsilon=min_gradient)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X: s, self.y: y, self.actions: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def get_action(self, sess, state):
        if self.epsilon >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.predict(sess, adapt_state(state)))

        # Decay epsilon over time
        if self.epsilon > final_epsilon:
            self.epsilon -= self.epsilon_step

        return action

    def get_trained_action(self, state):
        action = np.argmax(self.predict(sess, adapt_state(state)))
        return action

def copy_model_parameters(estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    return update_ops

def create_memory(env):
    # Populate the replay memory with initial experience
    replay_memory = []

    frame = env.reset()
    state = get_initial_state(frame)

    for i in range(replay_memory_init_size):
        action = np.random.choice(np.arange(env.action_space.n))
        frame, reward, done, _ = env.step(action)

        next_state = np.append(state[1:, :, :], preprocess(frame), axis=0)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            frame = env.reset()
            state = get_initial_state(frame)
        else:
            state = next_state

    return replay_memory


def setup_summary():
    with tf.variable_scope("episode"):
        episode_total_reward = tf.Variable(0., name="EpisodeTotalReward")
        tf.summary.scalar('Total Reward', episode_total_reward)
        episode_avg_max_q = tf.Variable(0., name="EpisodeAvgMaxQ")
        tf.summary.scalar('Average Max Q', episode_avg_max_q)
        episode_duration = tf.Variable(0., name="EpisodeDuration")
        tf.summary.scalar('Duration', episode_duration)
        episode_avg_loss = tf.Variable(0., name="EpisodeAverageLoss")
        tf.summary.scalar('Average Loss', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all(scope="episode")
    return summary_placeholders, update_ops, summary_op


if __name__ == "__main__":
    from tqdm import tqdm

    env = gym.make(env_name)
    tf.reset_default_graph()

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(env, scope="q", summaries_dir=tensorboard_path)
    target_estimator = Estimator(env, scope="target_q")

    copy_model = copy_model_parameters(q_estimator, target_estimator)

    summary_placeholders, update_ops, summary_op = setup_summary()

    # The replay memory
    replay_memory = create_memory(env)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        q_estimator.summary_writer.add_graph(sess.graph)

        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(network_path)
        if latest_checkpoint:
            print("Loading model checkpoint %s...\n" % latest_checkpoint)
            saver.restore(sess, latest_checkpoint)

        total_t = sess.run(tf.train.get_global_step())

        for episode in tqdm(range(n_episodes)):
            if total_t % save_interval == 0:
                # Save the current checkpoint
                saver.save(tf.get_default_session(), network_path)

            frame = env.reset()
            state = get_initial_state(frame)

            total_reward = 0
            total_loss = 0
            total_q_max = 0

            for duration in itertools.count():    
                # Maybe update the target estimator
                if total_t % network_update_interval == 0:
                    sess.run(copy_model)

                action = q_estimator.get_action(sess, state)
                frame, reward, terminal, _ = env.step(action)

                processed_frame = preprocess(frame)
                next_state = np.append(state[1:, :, :], processed_frame, axis=0)

                reward = np.clip(reward, -1, 1)
                replay_memory.append(Transition(state, action, reward, next_state, terminal))
                if len(replay_memory) > replay_memory_size:
                    replay_memory.popleft()

                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                adapted_state = adapt_batch_state(next_states_batch)

                q_values_next = q_estimator.predict(sess, adapted_state)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, adapted_state)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = adapt_batch_state(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

                total_q_max += np.max(q_values_next)
                total_loss += loss
                total_t += 1
                total_reward += reward
                if terminal:
                    break

            stats = [total_reward, total_q_max / duration, duration, total_loss / duration]
            for i in range(len(stats)):
                sess.run(update_ops[i], feed_dict={
                    summary_placeholders[i]: float(stats[i])
                })
            summary_str = sess.run(summary_op, )
            q_estimator.summary_writer.add_summary(summary_str, episode)
                
            env.env.ale.saveScreenPNG(six.b('%s/test_image_%05i.png' % (CHART_DIR, episode)))
            
        # Save the last checkpoint
        saver.save(tf.get_default_session(), network_path)
