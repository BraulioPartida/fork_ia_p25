import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        replay_capacity=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network y Target Network
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_capacity)

        # Hiperparámetros
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.target_update_freq = target_update_freq

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states_tensor).gather(1, actions_tensor)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            q_target = rewards_tensor + self.gamma * q_next * (1 - dones_tensor)

        loss = F.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Actualizar Target Network
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()