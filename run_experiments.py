import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from qlearning import QLearningAgent
from sarsa import SarsaAgent
from taxi import play_and_train

def run_experiment(agent_class, env, num_episodes=1000):
    env = RecordVideo(env, f"videos/{agent_class.__name__}/", 
                      episode_trigger=lambda episode_id: episode_id % 100 == 0)
    
    agent = agent_class(learning_rate=0.1, epsilon=0.1, gamma=0.99, legal_actions=list(range(env.action_space.n)))
    rewards = []

    for episode in range(num_episodes):
        reward = play_and_train(env, agent)
        rewards.append(reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")

    return rewards

# Run experiments
env = gym.make("Taxi-v3", render_mode="rgb_array")

q_learning_rewards = run_experiment(QLearningAgent, env)
sarsa_rewards = run_experiment(SarsaAgent, env)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1000), q_learning_rewards, label='Q-Learning')
plt.plot(range(1000), sarsa_rewards, label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Q-Learning vs SARSA Learning Curves')
plt.legend()
plt.savefig('learning_curves.png')
plt.show()

# Close the environment
env.close()