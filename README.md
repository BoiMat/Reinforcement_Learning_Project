# Reinforcement Learning Project

This repository contains the source code for the reinforcement learning project focused on Pokemon battles. The project includes custom environments, a Pokemon class, a multi-armed bandit class, and a Q-learning class. Additionally, there are two Jupyter notebooks showcasing tests for the multi-armed bandit and Q-learning algorithms.

## Project Structure

The main components of the project can be found in the `pokemon_gym` folder:

- **Environments (`envs.py`):** Custom Gym environments for Pokemon battles.
- **Pokemon Class (`pokemon.py`):** Implementation of the Pokemon class, representing the agents in the battles.
- **Multi-Armed Bandit Class (`mab.py`):** Implementation of the multi-armed bandit algorithm, adopting an $\epsilon$-greedy strategy.
- **Q-Learning Class (`qlearning.py`):** Implementation of the Q-learning algorithm.

## Test Notebooks

Two Jupyter notebooks, namely `mab.ipynb` and `qlearning.ipynb`, are included to showcase different tests for the multi-armed bandit and Q-learning algorithms, respectively.

1. **Convergence with Different Rewards:** Evaluate the convergence of the multi-armed bandit algorithm with various reward structures.
- **Guided Convergence reward:** Reward the Pokemon if it consistently uses a super-effective move throughout the episode.
- **Final HP Test:** Evaluate the performance based on the remaining HP of the Pokemon at the end of the episode.
1. **Mean Episode Convergence Test:** Determine the mean number of episodes required for the algorithm to reach the optimal policy.
2. **Learning with a Better Opponent:** Assess how the agent learns when facing an opponent that chooses a super-effective move with a probability of 0.5 or a random move.
