"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
import warnings
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import os

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)

        total_reward += r
        s = next_s

        if done:
            break
        # END SOLUTION

    return total_reward


rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

#################################################
# Run experiments and create videos
#################################################




warnings.filterwarnings("ignore", category=UserWarning)

def run_experiment(agent_class, env, num_episodes=1000):
    if agent_class.__name__ == 'SarsaAgent':
        agent = agent_class(learning_rate=0.1, gamma=0.99, legal_actions=list(range(env.action_space.n)))
    else:
        agent = agent_class(learning_rate=0.1, epsilon=0.1, gamma=0.99, legal_actions=list(range(env.action_space.n)))
    
    rewards = []

    # Training loop
    for episode in tqdm(range(num_episodes), desc=f"Training {agent_class.__name__}"):
        reward = play_and_train(env, agent)
        rewards.append(reward)

    # Print summary statistics
    print(f"\n{agent_class.__name__} Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Final 100 Episodes Average Reward: {np.mean(rewards[-100:]):.2f}")

    # Create a directory for videos if it doesn't exist
    video_dir = f"videos/{agent_class.__name__}"
    os.makedirs(video_dir, exist_ok=True)

    # Create a new environment for the final video
    video_env = gym.make("Taxi-v3", render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_dir, 
                      name_prefix=f"{agent_class.__name__}_episode_{num_episodes}")

    # Record one episode with the trained agent
    play_and_train(video_env, agent)

    # Close the video environment
    video_env.close()

    return rewards

def smooth_rewards(rewards, window_size=50):
    """Apply smoothing to the rewards."""
    cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

env = gym.make("Taxi-v3", render_mode="rgb_array")

q_learning_rewards = run_experiment(QLearningAgent, env)
sarsa_rewards = run_experiment(SarsaAgent, env)
q_learning_eps_scheduling_rewards = run_experiment(QLearningAgentEpsScheduling, env)

# Plot results with improvements
plt.figure(figsize=(12, 6))

# Plot raw data with low alpha
plt.plot(q_learning_rewards, label='Q-Learning', alpha=0.3)
plt.plot(sarsa_rewards, label='SARSA', alpha=0.3)
plt.plot(q_learning_eps_scheduling_rewards, label='Q-Learning with Epsilon Scheduling', alpha=0.3)

# Plot smoothed data
window_size = 50
plt.plot(smooth_rewards(q_learning_rewards, window_size), label='Q-Learning (Smoothed)', linewidth=2)
plt.plot(smooth_rewards(sarsa_rewards, window_size), label='SARSA (Smoothed)', linewidth=2)
plt.plot(smooth_rewards(q_learning_eps_scheduling_rewards, window_size), 
         label='Q-Learning with Epsilon Scheduling (Smoothed)', linewidth=2)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Learning Curves Comparison (Smoothed)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
plt.show()

# Close the environment
env.close()

def smooth_rewards(rewards, window_size=50):
    """Apply smoothing to the rewards."""
    cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
