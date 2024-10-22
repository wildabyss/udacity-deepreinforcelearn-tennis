import numpy as np
import random
import copy
from collections import namedtuple, deque

from learner.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,  random_seed=0, lr_actor=1e-4, lr_critic=1e-4, weight_decay=1e-4, future_discount=0.99,
                 soft_update_rate=0.001, replay_buffer_size=int(1e6), replay_batch_size=int(120),
                 add_noise=True, use_two_mems=False, bad_mem_ratio=0.2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.seed = random.seed(random_seed)
        self.replay_batch_size = replay_batch_size
        self.future_discount = future_discount
        self.soft_update_rate = soft_update_rate
        self.add_noise = add_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, replay_buffer_size, replay_batch_size, random_seed, use_two_mems, bad_mem_ratio)
    
    def step(self, state, action, reward, next_state, done, perform_learn=True, uniform_sampling=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if perform_learn and len(self.memory) >= self.replay_batch_size:
            experiences = self.memory.sample(uniform_sampling)
            self.learn(experiences)

    def act(self, state, noise_sigma=0.1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if self.add_noise:
            action += self.noise.sample(noise_sigma)
            
        # return action
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            future_discount (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.future_discount * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.soft_update_rate*local_param.data + (1.0-self.soft_update_rate)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        # self.seed = random.seed(seed)
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, sigma=1):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + sigma * np.array([random.random() for i in range(len(x))])
        # Use Gaussian noise to ensure we can still rarely sample large steps
        dx = self.theta * (self.mu - x) + sigma*np.random.normal(loc=0, scale=1, size=len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, replay_buffer_size, replay_batch_size, seed, use_two_mems=False, bad_mem_ratio=0.2):
        """Initialize a ReplayBuffer object.
        Params
        ======
            replay_buffer_size (int): maximum size of buffer
        """
        self.action_size = action_size
        self.replay_batch_size = replay_batch_size
        self.memory = deque(maxlen=replay_buffer_size)
        self.bad_memory = deque(maxlen=replay_buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.use_two_mems = use_two_mems
        self.bad_mem_ratio = bad_mem_ratio
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if self.use_two_mems and reward < 0:
            self.bad_memory.append(e)
        else:
            self.memory.append(e)
    
    def sample(self, uniform_sampling=False):
        """Randomly sample a batch of experiences from memory."""

        if self.use_two_mems:
            num_bad = min(len(self.bad_memory), np.ceil(self.replay_batch_size*self.bad_mem_ratio).astype(np.uint32))
        else:
            num_bad = 0

        # Choose from bad experience memory
        if uniform_sampling:
            experiences = random.sample(self.memory + self.good_memory + self.bad_memory, k=self.replay_batch_size)
        else:
            experiences = random.sample(self.memory, k=self.replay_batch_size-num_bad)
            if num_bad > 0:
                # Choose from bad experience memory
                experiences += random.sample(self.bad_memory, k=num_bad)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory) + len(self.bad_memory)