# TD Learning
"""
In this example, the agent explores the grid randomly (random policy) and updates the estimated value of each state based on the TD(0) formula. This approach illustrates how an agent can learn from every single step by bootstrapping (updating its estimate based on other estimates). It shows the basics of TD Learning, where the "temporal difference" refers to the difference between the estimated values of the current state and the next state, adjusted by the received reward. This example can be expanded with more sophisticated policies and learning algorithms to solve more complex reinforcement learning problems.
"""

import numpy as np

# Gridworld settings
grid_size = (5, 5)
termination_state = (4, 4)
start_state = (0, 0)
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
gamma = 0.99  # discount factor
alpha = 0.1  # learning rate
num_episodes = 1000

# Initialize the value function
V = np.zeros(grid_size)

def step(state, action):
    """Take a step in the environment."""
    next_state = (max(0, min(grid_size[0] - 1, state[0] + action[0])),
                  max(0, min(grid_size[1] - 1, state[1] + action[1])))
    reward = -1 if next_state != termination_state else 0
    return next_state, reward

def td_learning(V, episodes):
    """TD(0) Learning."""
    for _ in range(episodes):
        state = start_state
        while state != termination_state:
            action = actions[np.random.choice(len(actions))]  # Choose an action randomly
            next_state, reward = step(state, action)
            # TD(0) Update
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

td_learning(V, num_episodes)

print("Value function after TD(0) Learning:")
print(V)
