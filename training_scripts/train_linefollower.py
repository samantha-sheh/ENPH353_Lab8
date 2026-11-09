#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym  # RL library
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter  # Logging and visualizing data
import os
import pickle

def main():
    writer = SummaryWriter()  # Log training process
    env = gym.make('LineFollowerEnv-v0')  # Create environment w/ specified ID

    observation_space = env.observation_space  # Fetch obs space; discrete states that robot sees
    action_space = env.action_space  # Fetch action space; available actions

    epsilon = 0.8  # Exploration rate, takes random action with probability epsilon
    epsilon_decay = 0.99  # Decay rate

    q_table = np.zeros((observation_space.n, action_space.n))  # Q table of 0s w/ obs rows and action cols 

    highest_reward = 0  # Initial zero reward

    try:  # Handle interruptions gracefully
        state, _ = env.reset()
        for episode in range(1000):
            # Reset reward, action count, and done status each episode
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1

                # Choose action based on state and Q-table using epsilon-greedy strategy
                action = choose_action(state, q_table, epsilon, action_space)

                # Take a step in the environment w/ chosen action
                next_state, reward, terminated, truncated, _ = env.step(action) 
                done = terminated or truncated

                # Upd Q-table
                q_table = learn(state, action, next_state, reward, q_table)

                # Upd state and tot reward
                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            # Upd highest reward
            if total_reward > highest_reward:
                highest_reward = total_reward
                print(f"New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table")

            # Reset env and decay the exploration rate
            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    """
    Update Q values using the Bellman update equation.
    """
    alpha = 0.1  # Learning rate, higher faster but can be more unstable
    gamma = 0.9  # Discount factor, higher more long-term reward

    old_q = q_table[state, action]
    next_max_q = np.max(q_table[next_state])
    target = reward + gamma * next_max_q
    q_table[state, action] = old_q + alpha * (target - old_q)

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    """
    Select an action using an epsilon-greedy algorithm.
    """
    if random.random() < epsilon: 
        action = action_space.sample()  # Explore by choosing random action 0, 1, or 2
    else: 
        q_row = q_table[state]  # All Q vals associated with each action associated w/ state
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(q_row == max_q)
        if len(best_actions) == 0: 
            action = action_space.sample()  # If no best actions / all 0s, choose randomly
        else:
            action = int(np.random.choice(best_actions))  # Choose one of the best actions
    
    return action


def saveQ(q_table, filename):
    """
    Save Q-table to a path as a binary pickle file and a CSV file.
    """
    pkl_path = f"{filename}.pkl"
    csv_path = f"{filename}.csv"

    with open(pkl_path, "wb") as f:
        pickle.dump(q_table, f)

    np.savetxt(csv_path, q_table, delimiter=",", fmt="%.4f")

    print(f"Policy saved to {pkl_path} (and {csv_path})")

    return

def loadQ(filename):
    """
    Loads the Q-table. Attempts to load the binary .pkl first,
    falling back to .csv if pickle is unavailable.
    """
    pkl_path = f"{filename}.pkl"
    csv_path = f"{filename}.csv"

    # Try loading binary pickle first
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            q_table = pickle.load(f)
        print(f"Loaded policy from {pkl_path}")
        return q_table

    # Fallback to CSV
    if os.path.exists(csv_path):
        q_table = np.loadtxt(csv_path, delimiter=",")
        print(f"Loaded policy from CSV fallback: {csv_path}")
        return q_table

    print("No saved policy found.")
    
    return


if __name__ == "__main__":
    main()