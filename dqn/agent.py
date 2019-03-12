import numpy as np
import random
import tensorflow as tf

from replay_buffer import ReplayBuffer
from model import QNetwork


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, sess, state_size, action_size, seed, arguments):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.learning_rate = arguments['lr']
        self.gamma = arguments['gamma']
        self.update_every = arguments['update_every']
        self.tau = arguments['tau']
        self.history_size = arguments['history_size']

        self.buffer_size = arguments['buffer_size']
        self.batch_size = arguments['batch_size']

        # Q-Network
        self.qnetwork_local = QNetwork('local_q', state_size, action_size, self.history_size)
        self.qnetwork_target = QNetwork('target_q', state_size, action_size, self.history_size)

        copy_ops = []
        for local_w, target_w in zip(self.qnetwork_local.variables, self.qnetwork_target.variables):
            copy_op = tf.assign(local_w, local_w * self.tau + (1.0 - self.tau) * target_w)
            copy_ops.append(copy_op)

        self.copy_ops = tf.group(*copy_ops, name='copy_op')

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        q_values = self.sess.run(
                            self.qnetwork_local.q_values, 
                            feed_dict={
                                self.qnetwork_local.inputs: [state]
                            }).ravel().tolist()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(q_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        q_target_values = self.sess.run(
                            self.qnetwork_target.q_values, 
                            feed_dict={
                                self.qnetwork_target.inputs: next_states
                            })
        Q_targets_next = np.max(q_target_values, axis=1).reshape(-1, 1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        try:
            self.sess.run(self.qnetwork_local.optimizer, 
                            feed_dict={
                                self.qnetwork_local.learning_rate: self.learning_rate,
                                self.qnetwork_local.inputs: states,
                                self.qnetwork_local.actions: actions,
                                self.qnetwork_local.target_Q: np.squeeze(Q_targets),
                            })
        except:
            print(states.shape)
            print(actions.shape)
            print(rewards.shape)
            print(next_states.shape)
            print(dones.shape)
            print(q_target_values.shape)
            print(Q_targets_next.shape)
            print(Q_targets.shape)
            import sys
            sys.exit()

        # ------------------- update target network ------------------- #
        self.soft_update()                     

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        self.sess.run(self.copy_ops)

