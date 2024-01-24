import numpy as np

class ContextualEpsilonGreedyBandit:
    def __init__(self, num_arms, num_context, epsilon):
        self.num_arms = num_arms
        self.num_context = num_context
        self.epsilon = epsilon
        self.q_values = np.zeros((num_arms, *num_context))
        self.action_counts = np.zeros(num_arms)

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random arm
            return np.random.randint(self.num_arms)
        else:
            # Exploitation: choose the arm with the highest estimated value for the given context
            q_values_for_context = self.q_values[:, *context]
            return np.argmax(q_values_for_context)

    def update_q_values(self, action, context, reward):
        # Update Q-value estimate
        self.action_counts[action] += 1
        self.q_values[action, *context] += (reward - self.q_values[action, *context]) / self.action_counts[action]