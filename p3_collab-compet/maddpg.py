import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

LEARN_UPDATES = 2       # number of learning updates
UPDATE_EVERY = 2        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:

    def __init__(self, state_size, action_size, num_agents, params, seed=0):
        """Initialize a multi-agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            params (dict): hyperparameters
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.params = params
        random.seed(seed)
        
        # Replay memory
        self.memory = ReplayBuffer(self.params['buffer_size'], self.params['batch_size'], seed)
        # Multi-agents
        self.agents = [Agent(state_size, action_size, params, self.memory, seed) for i in range(num_agents)]
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory and use random sample from buffer to learn."""
        # Save experiences
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        # Sample and learn
        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        """Returns actions for each agent for given state as per current policy."""
        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i,:] = agent.act(states[i], add_noise)
        return actions

    def save_weights(self):
        """Save weights for each agent."""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(i+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(i+1))
    
    def reset(self):
        """Reset noise."""
        for agent in self.agents:
            agent.reset()

class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, memory, seed=0):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            params (dict): hyperparameters
            memory (ReplayBuffer): shared replay buffer
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.params['lr_actor'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.params['lr_critic'],
                                           weight_decay=self.params['weight_decay'])

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self):
        """Use random sample from buffer to learn."""
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.params['batch_size']:
                for i in range(LEARN_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.params['gamma'])

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        self.update_critic(states, actions, rewards, next_states, dones, gamma)
        # Update actor
        self.update_actor(states)
        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, self.params['tau'])
        self.soft_update(self.actor_local, self.actor_target, self.params['tau'])

    def update_actor(self, states):
        """Update actor parameters using given batch of experience tuples."""
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def update_critic(self, states, actions, rewards, next_states, dones, gamma):
        """Update critic parameters using given batch of experience tuples."""
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = self.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = self.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = self.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = self.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))

        return (states, actions, rewards, next_states, dones)

    def convert_to_tensor(self, array):
        """Convert numpy array to tensor"""
        return torch.from_numpy(array).float().to(device)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)