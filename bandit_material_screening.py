import numpy as np
import matplotlib.pyplot as plt


class MaterialScreeningEnv:
    
    def __init__(self, means: np.ndarray, noise_std: float = 0.05):
        self.means = np.array(means)
        self.k = len(means)
        self.noise_std = noise_std
        self.optimal_mean = np.max(self.means)
        self.optimal_arm = int(np.argmax(self.means))

    def step(self, action: int) -> float:
        noise = np.random.normal(0, self.noise_std)
        return float(self.means[action] + noise)


class UCB1Agent:
   
    def __init__(self, n_arms: int, c: float = np.sqrt(2)):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)
        self.q_values = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0

    def select_action(self) -> int:
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        bonus = self.c * np.sqrt(np.log(self.total_pulls) / self.counts)
        return int(np.argmax(self.q_values + bonus))

    def update(self, action: int, reward: float):
        self.total_pulls += 1
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]


class RandomScreeningAgent:
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)

    def select_action(self) -> int:
        return int(np.random.randint(0, self.n_arms))

    def update(self, action: int, reward: float):
        self.counts[action] += 1


def run_experiment(env: MaterialScreeningEnv, agent_class, horizon: int, trials: int, **agent_kwargs):
    
    rewards = np.zeros((trials, horizon))
    cumulative_regret = np.zeros((trials, horizon))

    for t in range(trials):
        agent = agent_class(n_arms=env.k, **agent_kwargs)
        trial_regret = 0.0

        for step in range(horizon):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)

            trial_regret += (env.optimal_mean - env.means[action])
            rewards[t, step] = reward
            cumulative_regret[t, step] = trial_regret

    return rewards, cumulative_regret


def moving_average(data: np.ndarray, window_size: int = 30) -> np.ndarray:
      return np.convolve(data, np.ones(window_size), 'valid') / window_size


def main():
    np.random.seed(42)

    true_means = np.linspace(0.10, 0.95, num=20)
    T = 1000
    N_TRIALS = 100

    env = MaterialScreeningEnv(means=true_means, noise_std=0.05)
    print(f"Optimal Candidate Index: {env.optimal_arm} (True Mean Yield: {env.optimal_mean:.2f})")

    print(f"Running {N_TRIALS} trials of UCB1 (c = sqrt(2))...")
    ucb_rewards, ucb_regret = run_experiment(env, UCB1Agent, horizon=T, trials=N_TRIALS, c=np.sqrt(2))

    print(f"Running {N_TRIALS} trials of Random Baseline...")
    rnd_rewards, rnd_regret = run_experiment(env, RandomScreeningAgent, horizon=T, trials=N_TRIALS)

    steps = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ucb_reg_mean = np.mean(ucb_regret, axis=0)
    rnd_reg_mean = np.mean(rnd_regret, axis=0)
    ucb_reg_se = np.std(ucb_regret, axis=0) / np.sqrt(N_TRIALS)
    rnd_reg_se = np.std(rnd_regret, axis=0) / np.sqrt(N_TRIALS)

    axes[0].plot(steps, ucb_reg_mean, label='UCB1 (c = √2)', color='#1f77b4', lw=2)
    axes[0].fill_between(steps, ucb_reg_mean - ucb_reg_se, ucb_reg_mean + ucb_reg_se, color='#1f77b4', alpha=0.2)
    axes[0].plot(steps, rnd_reg_mean, label='Random Baseline', color='#d62728', linestyle='--', lw=2)
    axes[0].fill_between(steps, rnd_reg_mean - rnd_reg_se, rnd_reg_mean + rnd_reg_se, color='#d62728', alpha=0.2)
    axes[0].set_title('Cumulative Regret vs. Steps', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Screening Rounds (t)', fontsize=11)
    axes[0].set_ylabel('Expected Cumulative Regret', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    w = 30
    ucb_avg_r = np.mean(ucb_rewards, axis=0)
    rnd_avg_r = np.mean(rnd_rewards, axis=0)
    smooth_steps = np.arange(w, T + 1)

    axes[1].plot(smooth_steps, moving_average(ucb_avg_r, w), label='UCB1 (Rolling Mean)', color='#1f77b4', lw=2)
    axes[1].plot(smooth_steps, moving_average(rnd_avg_r, w), label='Random Baseline', color='#d62728', linestyle='--', lw=2)
    axes[1].axhline(y=env.optimal_mean, color='green', linestyle=':', label=f'Optimal Material ({env.optimal_mean:.2f})')
    axes[1].set_title('Screening Reward Convergence', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Screening Rounds (t)', fontsize=11)
    axes[1].set_ylabel('Observed Yield', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = 'material_screening_learning_curves.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Comparison curves saved to '{plot_filename}'.")


if __name__ == "__main__":
    main()
