import random
from typing import List, Dict

class BaselineAgent:
    """Rule-based baseline agent for comparison"""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def act(self, state_vector: List[float]) -> int:
        """Simple rule-based policy"""
        # Extract state information
        fatigue = state_vector[-3] if len(state_vector) > 3 else 0
        focus_streak = state_vector[-2] if len(state_vector) > 2 else 0
        
        # Rule-based decision making
        if fatigue > 0.7:
            # Take break when tired
            return 1  # take_break
        elif focus_streak > 50:
            # Continue studying when in flow
            return 0  # study
        elif random.random() < 0.3:
            # Random exploration
            return random.randint(0, self.action_dim - 1)
        else:
            # Default to study
            return 0
    
    def train(self):
        """Baseline agent doesn't train"""
        pass
    
    def remember(self, state, action, reward, next_state, done):
        """Baseline agent doesn't remember"""
        pass