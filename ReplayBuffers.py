"""
ReplayBuffers.py

Replay buffer implementations used across the project for different RL algorithms.

Contents
--------
- `DT_ReplayBuffer`: trajectory buffer for Decision Transformer training (pads to max length).
- `POMO_ReplayBuffer`: simple buffer for POMO-style batched rollouts (stores state/action/reward/mask).
- `ReplayBuffer`: uniform (or recent) experience replay with n-step return support.
- `PrioritizedReplay`: classic Prioritized Experience Replay (PER) with proportional prioritization.
- `N_Steps_Prioritized_ReplayBuffer`: PER using a SumTree, with n-step returns and IS weights.
- `SumTree`: binary indexed tree storing priorities for efficient sampling.

Import safety
-------------
This module defines classes only and has no top-level execution block.
"""

import numpy as np 
from collections import deque, namedtuple
import random
import torch

class DT_ReplayBuffer:
    """
    Trajectory replay buffer for Decision Transformer training with padding and masks.
    
    Parameters
    ----------
    See signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, trajectory):
        """
        Add one transition (or trajectory) to the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.memory.append(trajectory)

    def sample(self, max_len=None):
        """
        Sample a batch from the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        batch = random.sample(self.memory, self.batch_size)

        if max_len is None:
            max_len = max(len(traj[0]) for traj in batch)

        state_batch, action_batch, reward_batch, timestep_batch = [], [], [], []

        for states, actions, returns, timesteps in batch:
            traj_len = states.size(0)

            pad_len = max_len - traj_len
            state_batch.append(torch.cat([states, torch.zeros(pad_len, *states.shape[1:])]))
            action_batch.append(torch.cat([actions, torch.zeros(pad_len, dtype=torch.long)]))
            reward_batch.append(torch.cat([returns, torch.zeros(pad_len, 1)]))
            timestep_batch.append(torch.cat([timesteps, torch.zeros(pad_len, dtype=torch.long)]))

        return (
            torch.stack(state_batch),
            torch.stack(action_batch),
            torch.stack(reward_batch),
            torch.stack(timestep_batch),
            torch.tensor([[1]*len_ + [0]*(max_len - len_) for len_ in [s.size(0) for s, *_ in batch]], dtype=torch.bool)
        )

    def __len__(self):
        """
        Return the number of stored items.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return len(self.memory)


class POMO_ReplayBuffer:
    """
    Replay buffer for POMO rollouts storing (state, action, reward, mask) tensors.
    
    Parameters
    ----------
    See signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, mask):
        """
        Add one transition (or trajectory) to the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.memory.append((state, action, reward, mask))

    def sample(self):
        """
        Sample a batch from the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        batch = random.sample(self.memory,self.batch_size)
        states, actions, rewards, masks = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        masks = torch.stack(masks)
        return (states, actions, rewards, masks)

    def __len__(self):
        """
        Return the number of stored items.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return len(self.memory)


class ReplayBuffer:

    """
    Basic replay buffer with optional recent sampling and n-step return computation.
    
    Parameters
    ----------
    See signature.
    
    Returns
    -------
    See implementation.
    """

    def __init__(self, buffer_size, batch_size, seed, gamma, n_steps, rdm):

        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.rdm = rdm
    
    def add(self, state, action, reward, next_state, done):

        """
        Add one transition (or trajectory) to the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self, n_step_buffer):
        """
        Compute the discounted n-step return from a buffer of transitions.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        Return = 0
        for idx in range(self.n_steps):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]
               
    def sample(self):
        
        """
        Sample a batch from the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        
        if self.rdm:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            experiences = list(self.memory)[-self.batch_size:]
        
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = np.stack(states)
        next_states = np.stack(next_states)
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplay(object):

    """
    Prioritized Experience Replay (proportional) with importance-sampling weights.
    
    Parameters
    ----------
    See signature.
    
    Returns
    -------
    See implementation.
    """

    def __init__(self, buffer_size, batch_size, seed, gamma, n_steps, alpha=0.6, beta_start = 0.4, beta_frames=63000):
        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.buffer_size  = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.pos        = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.gamma = gamma

    def calc_multistep_return(self,n_step_buffer):
        """
        Compute the discounted n-step return from a buffer of transitions.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        Return = 0
        for idx in range(self.n_steps):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]
    
    def beta_by_frame(self, frame_idx):

        """
        Compute annealed beta value for importance sampling weights.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        
        """
        Add one transition (or trajectory) to the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
            e = self.experience(state, action, reward, next_state, done)
            max_prio = self.priorities.max() if self.memory else 1.0 # gives max priority if buffer is not empty else 1

            if len(self.memory) < self.buffer_size:
                self.memory.append(e)
            else:
                self.memory[self.pos] = e

            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.buffer_size # lets the pos circle in the ranges of buffer_size if pos+1 > cap --> new posi = 0
        
        
    def sample(self):
        """
        Sample a batch from the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        N = len(self.memory)
        if N == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos] #### CHECK PRIO
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.memory[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        
        states = np.stack(states)
        next_states = np.stack(next_states)
  
        return states, actions, rewards, next_states, dones, indices, weights  
    
    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update priorities for a set of sampled indices.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        for idx, prio in zip(batch_indices, abs(batch_priorities)):
            self.priorities[idx] = prio 

    def __len__(self):
        """
        Return the number of stored items.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return len(self.memory)
    
class N_Steps_Prioritized_ReplayBuffer():
    """
    n-step prioritized replay buffer backed by a SumTree.
    
    Parameters
    ----------
    See signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, buffer_size, batch_size, seed, gamma, n_steps, alpha=0.6, beta_start = 0.4):
        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = 80000
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.buffer_size  = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = np.random.seed(seed)
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.gamma = gamma      
        self.sum_tree = SumTree(self.buffer_size)
        self.current_size = 0
        self.pos = 0
        
    def calc_multistep_return(self,n_step_buffer):
        """
        Compute the discounted n-step return from a buffer of transitions.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        Return = 0
        for idx in range(self.n_steps):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]
    
    def beta_by_frame(self, frame_idx):

        """
        Compute annealed beta value for importance sampling weights.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        return min(1.0, self.beta_start + (1.0 - self.beta_start) * frame_idx / self.beta_frames)
            
    def add(self, state, action, reward, next_state, done):

        """
        Add one transition (or trajectory) to the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            state, action, n_steps_reward, next_state, done = self.calc_multistep_return(self.n_step_buffer) 
            e = self.experience(state, action, reward, next_state, done)
            priority = self.sum_tree.priority_max if self.memory else 1.0 # gives max priority if buffer is not empty else 1
            
            if len(self.memory) < self.buffer_size:
                self.memory.append(e)
            else:
                self.memory[self.pos] = e
                        
            self.sum_tree.update(data_index=self.pos, priority=priority)  
            self.pos = (self.pos + 1) % self.buffer_size
            self.current_size = min(self.current_size + 1, self.buffer_size)
        

    def sample(self):
                        
        """
        Sample a batch from the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
                        
        indices, weights = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        
        self.beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        samples = [self.memory[idx] for idx in indices]
 
        states, actions, rewards, next_states, dones = zip(*samples) 
        
        states = np.stack(states)
        next_states = np.stack(next_states)
  
        return states, actions, rewards, next_states, dones, indices, weights  


    def update_priorities(self, batch_indices, td_errors):
        """
        Update priorities for a set of sampled indices.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for idx, prio in zip(batch_indices, abs(priorities)):
            self.sum_tree.update(data_index=idx, priority=prio)
    
            
    def __len__(self):
        """
        Return the number of stored items.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return len(self.memory)

    
class SumTree():
    """
    Story data with its priority in the tree.
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_size):
        """
        Initialize the buffer.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        self.buffer_size = buffer_size  
        self.tree_capacity = 2 * buffer_size - 1  
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index, priority):

        """
        Update a node priority and propagate the change up the SumTree.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """

        tree_index = data_index + self.buffer_size - 1  
        change = priority - self.tree[tree_index]  
        self.tree[tree_index] = priority  

        while tree_index != 0:  
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v):
        """
        Retrieve a leaf index given a cumulative priority mass.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        parent_idx = 0  
        while True:
            child_left_idx = 2 * parent_idx + 1  
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  
                tree_index = parent_idx  
                break
            else:  
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_size + 1  
        return data_index, self.tree[tree_index]  

    def get_batch_index(self, current_size, batch_size, beta):
        """
        Sample a batch of indices and compute IS weights using stratified segments.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        batch_index = np.zeros(batch_size, dtype=int)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.get_index(v)
            batch_index[i] = index
            prob = priority / self.priority_sum  
            IS_weight[i] = (current_size * prob) ** (-beta)
        IS_weight /= IS_weight.max()  # normalization

        return batch_index, IS_weight
  
    @property
    def priority_sum(self):
        """
        Total priority mass (root node).
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return self.tree[0]  

    @property
    def priority_max(self):
        """
        Maximum leaf priority currently stored.
        
        Parameters
        ----------
        See signature.
        
        Returns
        -------
        See implementation.
        """
        return self.tree[self.buffer_size - 1:].max()
