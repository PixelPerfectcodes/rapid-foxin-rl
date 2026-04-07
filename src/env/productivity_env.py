import numpy as np
import yaml
from typing import Dict, Tuple
from dataclasses import dataclass
import os
import random

@dataclass
class EnvState:
    current_state: str
    fatigue_level: float
    focus_streak: int
    attention_drift: float
    step_count: int

class ProductivityEnv:
    def __init__(self, config_path: str = "src/env/openenv.yaml"):
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.states = self.config.get('states', {})
        self.actions = self.config.get('actions', {})
        self.params = self.config.get('parameters', {})
        
        self.state_names = list(self.states.keys())
        self.action_names = list(self.actions.keys())
        self.last_reward_breakdown = {}
        
        self.reset()
    
    def _get_default_config(self):
        return {
            'states': {
                'focused': {'reward_range': [5, 15], 'transition': {'focused': 0.6, 'distracted': 0.2, 'tired': 0.15, 'deep_focus': 0.05}},
                'distracted': {'reward_range': [-15, -5], 'transition': {'focused': 0.4, 'distracted': 0.5, 'tired': 0.1, 'deep_focus': 0.0}},
                'tired': {'reward_range': [-10, -2], 'transition': {'focused': 0.3, 'distracted': 0.3, 'tired': 0.4, 'deep_focus': 0.0}},
                'deep_focus': {'reward_range': [15, 25], 'transition': {'focused': 0.3, 'distracted': 0.05, 'tired': 0.05, 'deep_focus': 0.6}}
            },
            'actions': {
                'study': {'effect': {'focused': 0.7, 'deep_focus': 0.2, 'distracted': 0.05, 'tired': 0.05}},
                'take_break': {'effect': {'focused': 0.5, 'tired': 0.3, 'distracted': 0.15, 'deep_focus': 0.05}},
                'use_phone': {'effect': {'distracted': 0.8, 'focused': 0.1, 'tired': 0.1, 'deep_focus': 0.0}},
                'switch_task': {'effect': {'focused': 0.5, 'distracted': 0.2, 'tired': 0.2, 'deep_focus': 0.1}}
            },
            'parameters': {
                'max_steps': 1000,
                'fatigue_accumulation_rate': 0.05,
                'burnout_threshold': 0.8,
                'focus_streak_bonus': 2.0,
                'time_decay_factor': 0.95,
                'attention_drift_rate': 0.1
            }
        }
    
    def reset(self) -> Dict:
        self.state = EnvState(
            current_state="focused",
            fatigue_level=0.0,
            focus_streak=0,
            attention_drift=0.0,
            step_count=0
        )
        return self.get_state()
    
    def get_state(self) -> Dict:
        state_vector = np.zeros(len(self.state_names) + 3)
        state_vector[self.state_names.index(self.state.current_state)] = 1
        state_vector[-3] = self.state.fatigue_level
        state_vector[-2] = min(1.0, self.state.focus_streak / 100)
        state_vector[-1] = self.state.attention_drift
        
        return {
            "vector": state_vector.tolist(),
            "current_state": self.state.current_state,
            "fatigue": self.state.fatigue_level,
            "focus_streak": self.state.focus_streak,
            "attention_drift": self.state.attention_drift
        }
    
    def step(self, action: str, ai_focus_score: float = 75.0) -> Tuple[Dict, float, bool, Dict]:
        if action not in self.actions:
            action = 'study'
        
        action_config = self.actions[action]
        
        # Get transition probabilities
        transition_probs = action_config['effect']
        
        # Apply AI adjustment
        ai_adjustment = (ai_focus_score - 50) / 100
        adjusted_probs = self._adjust_transitions(transition_probs, ai_adjustment)
        
        # Sample next state
        new_state = np.random.choice(
            list(adjusted_probs.keys()),
            p=list(adjusted_probs.values())
        )
        
        # Calculate reward
        reward = self._calculate_reward(new_state, action, ai_focus_score)
        
        # Update dynamics
        self._update_dynamics(new_state, action)
        
        # Update state
        self.state.current_state = new_state
        self.state.step_count += 1
        
        # Check if done
        done = self.state.step_count >= self.params['max_steps'] or \
               self.state.fatigue_level >= self.params['burnout_threshold']
        
        info = {
            "reward_breakdown": self.last_reward_breakdown,
            "state_transition": f"{self.state.current_state} -> {new_state}"
        }
        
        return self.get_state(), reward, done, info
    
    def _adjust_transitions(self, probs: Dict, ai_adjustment: float) -> Dict:
        adjusted = probs.copy()
        if ai_adjustment > 0:
            adjusted['focused'] = min(1.0, adjusted.get('focused', 0) + ai_adjustment * 0.2)
            adjusted['deep_focus'] = min(1.0, adjusted.get('deep_focus', 0) + ai_adjustment * 0.1)
        else:
            adjusted['distracted'] = min(1.0, adjusted.get('distracted', 0) + abs(ai_adjustment) * 0.2)
        
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}
    
    def _calculate_reward(self, new_state: str, action: str, ai_score: float) -> float:
        reward = 0.0
        breakdown = {}
        
        # Base reward from state
        state_reward_range = self.states[new_state]['reward_range']
        base_reward = np.random.uniform(state_reward_range[0], state_reward_range[1])
        reward += base_reward
        breakdown['base'] = base_reward
        
        # AI bonus/penalty
        ai_bonus = (ai_score - 50) / 5
        reward += ai_bonus
        breakdown['ai_bonus'] = ai_bonus
        
        # Fatigue penalty
        fatigue_penalty = -self.state.fatigue_level * 10
        reward += fatigue_penalty
        breakdown['fatigue_penalty'] = fatigue_penalty
        
        # Focus streak bonus
        if new_state in ['focused', 'deep_focus']:
            streak_bonus = self.params['focus_streak_bonus'] * (self.state.focus_streak / 10)
            reward += streak_bonus
            breakdown['streak_bonus'] = streak_bonus
        
        # Consistency reward
        if new_state == self.state.current_state:
            reward += 2.0
            breakdown['consistency'] = 2.0
        
        # Time decay penalty
        time_decay = -self.state.step_count * self.params['time_decay_factor'] * 0.1
        reward += time_decay
        breakdown['time_decay'] = time_decay
        
        # Action-specific adjustments
        if action == 'study' and new_state in ['focused', 'deep_focus']:
            reward += 5.0
            breakdown['action_bonus'] = 5.0
        elif action == 'use_phone' and new_state == 'distracted':
            reward -= 10.0
            breakdown['action_penalty'] = -10.0
        
        self.last_reward_breakdown = breakdown
        return reward
    
    def _update_dynamics(self, new_state: str, action: str):
        # Update fatigue
        if new_state in ['deep_focus', 'focused']:
            self.state.fatigue_level += self.params['fatigue_accumulation_rate']
        elif new_state == 'tired':
            self.state.fatigue_level += self.params['fatigue_accumulation_rate'] * 2
        
        self.state.fatigue_level = min(1.0, self.state.fatigue_level)
        
        # Update focus streak
        if new_state in ['focused', 'deep_focus']:
            self.state.focus_streak += 1
        else:
            self.state.focus_streak = max(0, self.state.focus_streak - 2)
        
        # Update attention drift
        if action == 'use_phone':
            self.state.attention_drift += self.params['attention_drift_rate']
        else:
            self.state.attention_drift = max(0, self.state.attention_drift - 0.05)