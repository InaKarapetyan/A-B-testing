import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
from abc import ABC, abstractmethod

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MAB Application")

# Disable debug messages for matplotlib font searching
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

#________________________________________________________________________________________


class Bandit(ABC):
    """Abstract base class representing a multi-armed bandit algorithm."""
    
    @abstractmethod
    def __init__(self, p):
        self.p = p
    
    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, res):
        pass

    @abstractmethod
    def experiment(self, num_trials):
        pass

    @abstractmethod
    def report(self):
        pass
#________________________________________________________________________________________
    
    def store_rewards_to_csv(self, algorithm_name):
        filename = f'{algorithm_name}_rewards.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward'])
            for reward in self.rewards:
                writer.writerow([self.p, reward])
        logger.info(f'\033[92mRewards data for {algorithm_name} bandit (p={self.p}) has been stored in {filename}.\033[0m')
#________________________________________________________________________________________
        
    def calculate_cumulative_reward(self):
        self.cumulative_reward = sum(self.rewards)
#________________________________________________________________________________________
        
    def print_cumulative_reward(self):
        logger.info(f'\033[92mCumulative Reward for Bandit (p={self.p}): {self.cumulative_reward}\033[0m')

#________________________________________________________________________________________
        
class Visualization:
    """Class for visualizing bandit algorithms' performance."""
    def plot_learning_process(self, bandits, num_trials, title):
        plt.figure(figsize=(10, 5))
        for bandit in bandits:
            plt.plot(range(1, num_trials + 1), bandit.cumulative_average_rwrd, label=f'{bandit.__class__.__name__} Bandit {bandit.p:.2f}')

        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Average Reward')
        plt.legend()
        plt.show()

    def plot_epsilon_greedy(self, epsilon_greedy_bandits, num_trials, title):
        plt.figure(figsize=(10, 5))
        for bandit in epsilon_greedy_bandits:
            plt.plot(bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')

        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()

    def plot_thompson_sampling(self, thompson_bandits, num_trials, title):
        plt.figure(figsize=(10, 5))
        for bandit in thompson_bandits:
            plt.plot(bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')

        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()
    
    def plot_cumulative_rewards(self, epsilon_greedy_bandits, thompson_bandits, num_trials):
        plt.figure(figsize=(10, 5))

        # Plot cumulative rewards for Epsilon Greedy bandits
        for bandit in epsilon_greedy_bandits:
            cumulative_rewards = np.cumsum(bandit.rewards)
            plt.plot(range(1, num_trials + 1), cumulative_rewards, label=f'Epsilon Greedy Bandit {bandit.p:.2f}')

        # Plot cumulative rewards for Thompson Sampling bandits
        for bandit in thompson_bandits:
            cumulative_rewards = np.cumsum(bandit.rewards)
            plt.plot(range(1, num_trials + 1), cumulative_rewards, label=f'Thompson Sampling Bandit {bandit.p:.2f}')

        plt.title('Cumulative Rewards Comparison')
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Rewards')
        plt.legend()
        plt.show()

#________________________________________________________________________________________
        
def compare_cumulative_regret(epsilon_greedy_bandits, thompson_bandits, num_trials):
    """
    Compare the cumulative regret of Epsilon Greedy and Thompson Sampling algorithms.
    """
    plt.figure(figsize=(10, 5))
    for bandit in epsilon_greedy_bandits:
        plt.plot(range(1, num_trials + 1), bandit.cumulative_regret_dict.values(), label=f'Epsilon Greedy Bandit {bandit.p:.2f}')

    for bandit in thompson_bandits:
        plt.plot(range(1, num_trials + 1), bandit.cumulative_regret_dict.values(), label=f'Thompson Sampling Bandit {bandit.p:.2f}')

    plt.title('Comparison of Cumulative Regret')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.show()

#________________________________________________________________________________________
        
class EpsilonGreedy(Bandit):
    """Class representing the Epsilon Greedy bandit algorithm."""
    
    def __init__(self, true_mean, initial_epsilon):
        super().__init__(true_mean)
        self.p_estimate = 0
        self.N = 0
        self.rewards = []
        self.cumulative_regret = 0
        self.cumulative_regret_dict = {}
        self.cumulative_average_rwrd = []
        self.epsilon = initial_epsilon

    def __repr__(self):
        return f'EpsilonGreedy(true_mean={self.p}, initial_epsilon={self.epsilon})'

    def pull(self):
        return np.random.randn() + self.p

    def update(self, res):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + res) / self.N
        self.rewards.append(res)

    def experiment(self, num_trials):
        for trial in range(num_trials):
            if np.random.random() < self.epsilon:
                action = np.random.randn() + self.p
            else:
                action = self.p_estimate

            reward = self.pull()
            self.update(reward)
            self.cumulative_regret += (self.p - action)
            self.cumulative_regret_dict[trial + 1] = self.cumulative_regret
            self.cumulative_average_rwrd.append(np.mean(self.rewards))

            # Decay epsilon by 1/t
            self.epsilon = 1 / (trial + 1)

    def report(self):
        logger.info(f"Epsilon Greedy Bandit Algorithm")
        logger.info(f"True Mean: {self.p}")
        logger.info(f"Estimated Mean: {self.p_estimate}")
        logger.info(f"Cumulative Regret: {self.cumulative_regret}")

    def print_cumulative_reward(self):
        logger.info(f'\033[92mCumulative Reward for Bandit (p={self.p}): {np.sum(self.rewards)}\033[0m')

    def print_cumulative_regret(self):
        logger.info(f'\033[92mCumulative Regret for Bandit (p={self.p}): {self.cumulative_regret}\033[0m')

#________________________________________________________________________________________
         
class ThompsonSampling(Bandit):
    """Class representing the Thompson Sampling bandit algorithm."""
    
    def __init__(self, true_mean):
        super().__init__(true_mean)
        self.alpha = 1
        self.beta = 1
        self.rewards = []
        self.cumulative_regret = 0
        self.cumulative_regret_dict = {}
        self.cumulative_average_rwrd = []
    
    def __repr__(self):
        return f'ThompsonSampling(true_mean={self.p})'

    def pull(self):
        return np.random.beta(self.alpha, self.beta)

    def update(self, res):
        if res == 1:
            self.alpha += 1
        else:
            self.beta += 1
        self.rewards.append(res)

    def experiment(self, num_trials):
        for trial in range(num_trials):
            action = self.pull()
            reward = np.random.randn() + self.p
            self.update(reward)
            self.cumulative_regret += (self.p - action)
            self.cumulative_regret_dict[trial + 1] = self.cumulative_regret
            self.cumulative_average_rwrd.append(np.mean(self.rewards))

    def report(self):
        logger.info(f"Thompson Sampling Bandit Algorithm")
        logger.info(f"True Mean: {self.p}")
        logger.info(f"Cumulative Regret: {self.cumulative_regret}")

    def print_cumulative_reward(self):
        logger.info(f'\033[92mCumulative Reward for Bandit (p={self.p}): {np.sum(self.rewards)}\033[0m')

    def print_cumulative_regret(self):
        logger.info(f'\033[92mCumulative Regret for Bandit (p={self.p}): {self.cumulative_regret}\033[0m')
    
#________________________________________________________________________________________
        
if __name__ == "__main__":
    # Simulation parameters
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    initial_epsilon = 0.1  # Set initial epsilon value

    # Instantiate Epsilon Greedy bandits with initial epsilon value
    epsilon_greedy_bandits = [EpsilonGreedy(reward, initial_epsilon) for reward in bandit_rewards]

    # Instantiate Thompson Sampling bandits
    thompson_bandits = [ThompsonSampling(reward) for reward in bandit_rewards]

    # Perform experiments for both Epsilon Greedy and Thompson Sampling bandits
    for bandit_type in [epsilon_greedy_bandits, thompson_bandits]:
        for bandit in bandit_type:
            bandit.experiment(num_trials)
            bandit.report()

    # Instantiate Visualization object
    visualization = Visualization()

    # Plot the performance of Epsilon Greedy bandits
    visualization.plot_epsilon_greedy(epsilon_greedy_bandits, num_trials, 'Epsilon Greedy')

    # Plot the performance of Thompson Sampling bandits
    visualization.plot_thompson_sampling(thompson_bandits, num_trials, 'Thompson Sampling')

    # Plot the learning process of Epsilon Greedy bandits
    visualization.plot_learning_process(epsilon_greedy_bandits, num_trials, 'Learning Process: Epsilon Greedy')

    # Plot the learning process of Thompson Sampling bandits
    visualization.plot_learning_process(thompson_bandits, num_trials, 'Learning Process: Thompson Sampling')

    # Plot cumulative rewards for both Epsilon Greedy and Thompson Sampling bandits
    visualization.plot_cumulative_rewards(epsilon_greedy_bandits, thompson_bandits, num_trials)

    # Perform experiments for both Epsilon Greedy and Thompson Sampling bandits
    for bandit_type in [epsilon_greedy_bandits, thompson_bandits]:
        for bandit in bandit_type:
            bandit.experiment(num_trials)
            bandit.report()
            bandit.calculate_cumulative_reward()
            bandit.print_cumulative_reward()
            bandit.store_rewards_to_csv(algorithm_name=bandit.__class__.__name__)

# Compare cumulative regret
compare_cumulative_regret(epsilon_greedy_bandits, thompson_bandits, num_trials)



# BONUS
'''

- Make it Easier to Change Settings:
Put all the adjustable settings like the number of trials or initial epsilon at the top of the script. This way, users can easily find and change them.

- Make the Algorithms Faster:
Look for ways to speed up the bandit algorithms so they run faster. This could involve finding more efficient ways to calculate things or using better algorithms.

- Check if the Results Are Reliable:
Use statistical tools to check if the results you're getting are reliable or if they might just be due to chance.


'''

'''
just an example)
from scipy.stats import ttest_ind

# Collect data from bandit experiments
epsilon_greedy_rewards = [bandit.rewards for bandit in epsilon_greedy_bandits]
thompson_rewards = [bandit.rewards for bandit in thompson_bandits]

# Perform t-test to compare mean rewards between Epsilon Greedy and Thompson Sampling
t_statistic, p_value = ttest_ind(epsilon_greedy_rewards, thompson_rewards)

# Determine if the results are statistically significant
if p_value < 0.05:
    logger.info("The difference in performance between Epsilon Greedy and Thompson Sampling is statistically significant.")
else:
    logger.info("There is no statistically significant difference in performance between Epsilon Greedy and Thompson Sampling.")

    '''

# or smth like this 

'''
def calculate_confidence_interval(self, confidence=0.95):
        """
        Calculate the confidence interval for the estimated mean reward.

        Args:
            confidence (float): The desired level of confidence (default is 0.95 for 95% confidence).

        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        rewards_array = np.array(self.rewards)
        mean_reward = np.mean(rewards_array)
        stderr = scipy.stats.sem(rewards_array)
        margin_of_error = stderr * scipy.stats.t.ppf((1 + confidence) / 2.0, len(rewards_array) - 1)
        lower_bound = mean_reward - margin_of_error
        upper_bound = mean_reward + margin_of_error
        return lower_bound, upper_bound
'''