import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent


def train(env_name, num_episodes=500, max_steps=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []
    losses_history = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push_experience(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            ep_reward += reward

            if done:
                break

        rewards_history.append(ep_reward)
        if loss is not None:
            losses_history.append(loss)

        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {ep}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Visualizar recompensas
    plt.plot(rewards_history)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title(f'Training Rewards - {env_name}')
    plt.show()

    # Guardar modelo
    torch.save(agent.q_net.state_dict(), f"dqn_{env_name}.pth")


if __name__ == "__main__":
    train('CartPole-v1', num_episodes=500)