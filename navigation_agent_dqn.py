import numpy as np
import random
from collections import namedtuple, deque
import pdb 
from navigation_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, drop_p=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.drop_p = drop_p

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, drop_p=drop_p).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, drop_p=drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Convert state(numpy array of length 8) to a tensor with floating point entries.
        # This is achieved with the command torch.from_numpy(state).float(). 
        # Then, we convert the 1-D vector of 8 entries to a tensor of 2-D using unsqueeze(D)
        # where D is the inserted dimension (or axis).
        #pdb.set_trace()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set the network to evaluation mode.
        self.qnetwork_local.eval()
        # Deactive the gradients for the forward pass
        with torch.no_grad():
            # Get the probabilities for each action evaluated in the local network.
            action_values = self.qnetwork_local(state)
        # Set the network back to train mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Return the index of the action with largest value. the .cpu() method set the 
            # tensor to be manipualted in a cpu rather than cuda. The .data attribute gets the data of a 
            # Variable, but not required since it is already a tensor. Then we convert to numpy.
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Return a random action using epsilon-greedy policy
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Unpack the variables below from the experience tuple
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model.
        # .detach() --> turns off the gradient calculation.
        # .max(1) --> Gets the max Q value of the action space when we evaluate qnetwork_target
        #             with next states. (1) means get the max along dim = 1 (rows).
        #             the output is an array of two tensors, one with max values and the second 
        #             one with the indices of those values. [0] gets the values only (Q-value)
        # .unsqueeze(1) --> Add an extra axis to obtain a 2D array.
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between minibatches,
        # you have to zero them out at the start of a new minibatch. This is exactly like how a general (additive) 
        # accumulator variable is initialized to 0 in code.
        # By the way, the best practice is to use the zero_grad() function on the optimizer.
        self.optimizer.zero_grad()
        
        # Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. This is why you 
        # should call optimizer.zero_grad() after each .step() call. Note that following the first .backward call, a 
        # second call is only possible after you have performed another forward pass.
        loss.backward()
        
        # optimizer.step performs a parameter update based on the current gradient (stored in .grad attribute of a
        # parameter) and the update rule.
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)