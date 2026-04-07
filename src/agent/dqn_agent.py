import numpy as np
import random
from collections import deque
from typing import List, Tuple

class DQNAgent:
    """Simplified DQN Agent without PyTorch for compatibility"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Q-table for simple RL (tabular Q-learning)
        self.q_table = {}
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.alpha = learning_rate
        
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.losses = []
    
    def _get_state_key(self, state_vector: List[float]) -> str:
        """Convert state vector to hashable key"""
        # Discretize continuous values
        discretized = [round(x, 2) for x in state_vector]
        return ','.join(map(str, discretized))
    
    def act(self, state_vector: List[float], eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_key = self._get_state_key(state_vector)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_dim
        
        return int(np.argmax(self.q_table[state_key]))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self) -> float:
        """Train the agent using Q-learning update"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            # Initialize Q-values if not exist
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_dim
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_dim
            
            # Q-learning update
            old_value = self.q_table[state_key][action]
            next_max = max(self.q_table[next_state_key]) if not done else 0
            
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            self.q_table[state_key][action] = new_value
            
            total_loss += abs(new_value - old_value)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        avg_loss = total_loss / self.batch_size
        self.losses.append(avg_loss)
        
        return avg_loss
    
    def save(self, path: str):
        """Save Q-table to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def load(self, path: str):
        """Load Q-table from file"""
        import pickle
        import os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
            return True
        return False