# rl_engine.py
import numpy as np

class RLEnvironment:
    """
    MDP formulation for E-MARS reliability control [1].
    State:  module-level health vector [latency, error_rate, cpu, mttr]
    Actions: 0=scale_up, 1=scale_down, 2=isolate
    Reward: R = w1*(1/latency) + w2*(1-error_rate)
              + w3*(resource_eff) - w4*(scaling_cost)
    """
    def __init__(self, num_modules):
        self.num_modules = num_modules
        self.state_dim = num_modules * 4   # 4 features per module
        self.action_dim = 3                # scale_up, scale_down, isolate
        self.w = [0.3, 0.4, 0.2, 0.1]     # reward weights

    def get_state(self, health_scores, telemetry_df):
        state = []
        for module_id, info in health_scores.items():
            module_data = telemetry_df[
                telemetry_df["service"].isin(info["services"])
            ]
            if not module_data.empty:
                state.extend([
                    module_data["mean_latency"].mean(),
                    module_data["error_rate"].mean(),
                    module_data["cpu_utilization"].mean(),
                    1 - info["health"]  # inverse health as MTTR proxy
                ])
            else:
                state.extend([0, 0, 0, 0])
        return np.array(state, dtype=np.float32)

    def compute_reward(self, latency, error_rate,
                       resource_eff, scaling_cost):
        """Reward function R [1]"""
        return (self.w[0] * (1 / (latency + 1e-6)) +
                self.w[1] * (1 - error_rate) +
                self.w[2] * resource_eff -
                self.w[3] * scaling_cost)


class DQNAgent:
    """
    DQN for reactive short-horizon decisions [1].
    Handles immediate fault isolation and traffic rerouting.
    """
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.95, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((100, action_dim))  # simplified Q-table

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_idx = int(np.sum(state) * 10) % 100
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state, action, reward, next_state):
        state_idx = int(np.sum(state) * 10) % 100
        next_idx = int(np.sum(next_state) * 10) % 100
        td_target = reward + self.gamma * np.max(
            self.q_table[next_idx]
        )
        self.q_table[state_idx, action] += 0.01 * (
            td_target - self.q_table[state_idx, action]
        )


class PPOAgent:
    """
    PPO for strategic longer-horizon decisions [1].
    Handles gradual autoscaling and resource rebalancing.
    """
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.policy = np.ones(action_dim) / action_dim  # uniform init

    def select_action(self, state):
        """Sample action from policy distribution"""
        return np.random.choice(self.action_dim, p=self.policy)

    def update_policy(self, rewards):
        """Simplified PPO policy update"""
        if np.sum(rewards) > 0:
            self.policy = np.abs(rewards) / (np.sum(np.abs(rewards)) + 1e-6)