#!/usr/bin/env python3

# The code below was heavily inspired from "Deep Reinforcement Learning Hands-On" 
# by Maxim Lapan.

import torch
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim

import gym_gazebo
import gymnasium as gym

from collections import namedtuple
from torch.utils.tensorboard.writer import SummaryWriter


# Helper classes and functions
class NeuralNet(nn.Module): 
    '''
    @brief  Takes observations from the environment and outputs probabilities for
            actions to take
    '''
    def __init__(self, obs_size, hidden_size, n_actions):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)  # Note: no softmax here
        )

    # nn.Module has special __call__ method that intercepts and calls forward 
    # whenever the NeuralNet is called eg. net(obs_tensor)  
    def forward(self, obs):
        return self.net(obs)  # Returns logits, raw scores


def generate_batches(env, net, batch_size):
    '''
    @brief a generator function that generates batches of episodes that are
           used to train NN on
    @param env: environment handler - allows us to reset and step the simulation
    @param net: neural network we use to predict the next action
    @param batch_size: number of episodes to compile
    @retval batch: returns a batch of batch_size episodes (each episode contains
                  a list of observations and actions and the total reward for
                  the episode)
    '''
    batch = []  # List of finished episodes
    current_episode_reward = 0.0  # Total reward from the current episode
    current_step_list = []  # (obs, act) pairs from the current episode
    sm = nn.Softmax(dim=1) # Softmax object used to convert raw NN outputs to probabilities
    
    # INDEX:
        # 0: cart position (m)
        # 1: cart velocity (m/s)  
        # 2: pole angle (rad)
        # 3: pole angular velocity (rad/s)
    obs, _ = env.reset() # First observation
    print(obs)

    # Main iteration loop
    i=0
    while True:
        i += 1
        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32)  # Turn into a tensor to pass into the NN
        act_prob_tensor = sm(net(obs_tensor))  # Action probabilities w/ softmax

        # Unpack the output of the NN to extract the probabilities associated
        # with each action.
        # 1) Extract the data field from the NN output
        # 2) Convert the tensors from the data field into numpy array
        # 3) Extract the first element of the network output. This is where 
        #    the probability distribution are stored. The second element of the
        #    network output stores the gradient functions (which we don't use)
        # 4) Result is a 1D array like [0.3, 0.7] 
        act_prob_np = act_prob_tensor.data.numpy()[0]

        # Pick a random action weighted by the probabilities
        action = np.random.choice(len(act_prob_np), p=act_prob_np)

        # Run one simulation step using the action we sampled
        next_obs, reward, is_done, _, _ = env.step(action)
        #print("Step: {} | reward: {} | is_done: {}".format(i, reward, is_done))

        # Upd current reward and action list
        current_episode_reward += reward
        current_step_list.append(Step(observation=obs, action=action))

        # Resetting an episode
        if is_done:
            batch.append(Episode(reward=current_episode_reward, steps=current_step_list))
            
            # Resetting episode stats
            current_episode_reward = 0.0
            current_step_list = []
            next_obs, _ = env.reset()
            i = 0

            # Resetting and returning a batch
            if len(batch) == batch_size:
                yield batch  # Hands the batch to whoever is looping over this generator
                # Resumes here when caller asks for next batch
                batch = []
        
        obs = next_obs


def filter_batches(batch, percentile):
    '''
    @brief given a batch of episodes determine which are the "elite" 
           episodes in the top percentile of the batch based on the episode
           reward
    @param batch:
    @param percentile:
    @retval train_obs_v: observation associated with elite episodes
    @retval train_act_v: actions associated with elite episodes (mapped to 
                         observations above)
    @retval reward_bound: the threshold reward over which an episode is 
                          considered elite - used for monitoring progress
    @retval reward_mean: mean reward - used for monitoring progress
    '''

    # Extract the total reward for each episode
    rewards = list(map(lambda s: s.reward, batch))

    # Find the lowest reward allowed
    reward_cutoff = np.percentile(rewards, percentile)

    # Diagnostic variable to see how good each batch is
    reward_mean = float(np.mean(rewards))

    # Training sets
    train_obs = []
    train_act = []

    for example_episode in batch:
        # Ignore episode if it doesn't pass the cutoff
        if example_episode.reward < reward_cutoff:
            continue
        
        # Append the episode to the training set if it passes
        train_obs.extend(map(lambda step: step.observation, example_episode.steps))
        train_act.extend(map(lambda step: step.action, example_episode.steps))

    # Convert arrays to tensors
    train_obs_tensor = torch.FloatTensor(np.array(train_obs))
    train_act_tensor = torch.LongTensor(np.array(train_act))

    return train_obs_tensor, train_act_tensor, reward_cutoff, reward_mean


# Stores the total reward of the episode and the steps taken within it
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# Stores information about each step
Step = namedtuple('Step', field_names=['observation', 'action'])

# Main script
def main(args=None):
    writer = SummaryWriter()  # Tensorboard logger
    env = gym.make('CartpoleEnv-v0')  # Sim env

    time.sleep(2.0)  # Time for Gazebo to start up

    obs_size = env.observation_space.shape[0]  # Size of obs space, 4
    n_action = env.action_space.n  # How many discrete actions, 2

    hidden_size = 128  # Hidden neurons in one hidden layer
    batch_size = 16  # Episodes per iteration
    percentile = 70  # Only top 30% of episodes elite
    reward_threshold = 600  # Mean reward needed to be solved

    net = NeuralNet(obs_size, hidden_size, n_action)  # Instantiate NN
    objective = nn.CrossEntropyLoss()  # Loss function, expects raw logits because applies softmax internally
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)  # Weight adjustment algorithm, learning rate

    try:
        i = 0
        # Iterate through episode objects
        for iter_no, batch in enumerate(generate_batches(env, net, batch_size)):
            obs_tensor, action_tensor, reward_cutoff, reward_mean = filter_batches(batch, percentile)
            i += 1
            print("Got a batch! ", i)
            
            # Zero out gradients
            optimizer.zero_grad()

            # Calculate the predicted logits for each action in the elite episodes
            # Pass observations to the network, forward pass on each observation
            act_logits_tensor = net(obs_tensor)

            # Calculate the cross entropy loss between predicted actions and actual actions
            # (predicted_action, target_action)
            loss_tensor = objective(act_logits_tensor, action_tensor)

            # Train the neural network (once per batch)
            # Upd the gradients w/ back prop
            loss_tensor.backward()

            # Update the parameters
            optimizer.step()

            print("========== Iteration: {} -> Loss: {}".
                  format(iter_no, loss_tensor.item()))

            # Print batch statistics
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_tensor.item(), reward_mean, reward_cutoff))
            
            writer.add_scalar("loss", loss_tensor.item(), iter_no)
            writer.add_scalar("reward_bound", reward_cutoff, iter_no)
            writer.add_scalar("reward_mean", reward_mean, iter_no)

            # When reward is above the threshold, stop training
            if reward_mean > reward_threshold:
                print("Solved!")
                break
            
    except KeyboardInterrupt:
        print("Training interrupted by user!")

    finally:
        writer.close()
        env.close()


if __name__ == '__main__':
    main()
