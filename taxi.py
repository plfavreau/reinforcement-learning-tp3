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
from gym.wrappers import RecordVideo
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

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


def run_experiment(agent_class, env, num_episodes=1000):
    env = RecordVideo(env, f"videos/{agent_class.__name__}/", 
                      episode_trigger=lambda episode_id: episode_id % 100 == 0)
    
    # Check if the agent class is SarsaAgent
    if agent_class.__name__ == 'SarsaAgent':
        agent = agent_class(learning_rate=0.1, gamma=0.99, legal_actions=list(range(env.action_space.n)))
    else:
        agent = agent_class(learning_rate=0.1, epsilon=0.1, gamma=0.99, legal_actions=list(range(env.action_space.n)))
    
    rewards = []

    for episode in range(num_episodes):
        reward = play_and_train(env, agent)
        rewards.append(reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")

    return rewards

env = gym.make("Taxi-v3", render_mode="rgb_array")

q_learning_rewards = run_experiment(QLearningAgent, env)
sarsa_rewards = run_experiment(SarsaAgent, env)
q_learning_eps_scheduling_rewards = run_experiment(QLearningAgentEpsScheduling, env)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1000), q_learning_rewards, label='Q-Learning')
plt.plot(range(1000), sarsa_rewards, label='SARSA')
plt.plot(range(1000), q_learning_eps_scheduling_rewards, label='Q-Learning with Epsilon Scheduling')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Learning Curves Comparison')
plt.legend()
plt.savefig('learning_curves.png')
plt.show()

# Close the environment
env.close()
