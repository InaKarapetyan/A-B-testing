# Multi-Armed Bandit Algorithms

This project implements two popular multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling. These algorithms are commonly used in reinforcement learning and online decision-making scenarios.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Multi-armed bandit algorithms are a class of algorithms used for sequential decision-making problems where an agent must decide between exploring new options and exploiting the best-known option. This project provides implementations of two such algorithms:

1. **Epsilon Greedy**: This algorithm balances exploration and exploitation by choosing the best-known option with probability 1 - ε and exploring with probability ε.
   
2. **Thompson Sampling**: This algorithm uses a Bayesian approach to estimate the probability distribution of rewards for each option. It samples from these distributions to decide which option to select.

## Installation

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the multi-armed bandit algorithms, follow these steps:

1. Import the necessary modules:

```python
import numpy as np
import matplotlib.pyplot as plt
from bandit import EpsilonGreedy, ThompsonSampling, compare_cumulative_regret
```

2. Instantiate the bandit algorithms with the desired parameters:

```python
epsilon_greedy_bandits = [EpsilonGreedy(reward, epsilon) for reward in bandit_rewards]
thompson_bandits = [ThompsonSampling(reward) for reward in bandit_rewards]
```

3. Run experiments and visualize results:

```python
# Run experiments
for bandit_type in [epsilon_greedy_bandits, thompson_bandits]:
    for bandit in bandit_type:
        bandit.experiment(num_trials)

# Visualize results
compare_cumulative_regret(epsilon_greedy_bandits, thompson_bandits, num_trials)
```

## Algorithm Overview

### Epsilon Greedy Algorithm

The Epsilon Greedy algorithm selects the best-known option with probability \(1 - \epsilon\) and explores other options with probability \(\epsilon\). This balance between exploration and exploitation ensures that the algorithm converges to the optimal solution while still exploring new options.

### Thompson Sampling Algorithm

Thompson Sampling uses a Bayesian approach to estimate the probability distribution of rewards for each option. It samples from these distributions to decide which option to select. This algorithm tends to perform well in a wide range of scenarios and is particularly effective when the rewards are not known beforehand.
