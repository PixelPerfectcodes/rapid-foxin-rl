from typing import Dict, List, Optional
import random

class TaskSystem:
    def __init__(self):
        self.tasks = {
            'easy': {
                'distraction_probability': 0.2,
                'fatigue_rate': 0.03,
                'reward_sensitivity': 1.2,
                'base_reward': 10
            },
            'medium': {
                'distraction_probability': 0.4,
                'fatigue_rate': 0.05,
                'reward_sensitivity': 1.0,
                'base_reward': 15
            },
            'hard': {
                'distraction_probability': 0.6,
                'fatigue_rate': 0.08,
                'reward_sensitivity': 0.8,
                'base_reward': 20
            }
        }
        self.current_difficulty = 'medium'
        self.task_progress = 0
        self.task_completion_threshold = 100
    
    def set_difficulty(self, difficulty: str):
        if difficulty in self.tasks:
            self.current_difficulty = difficulty
    
    def get_task_config(self) -> Dict:
        return self.tasks[self.current_difficulty]
    
    def update_progress(self, focus_score: float, action: str) -> float:
        config = self.tasks[self.current_difficulty]
        
        if action == 'study':
            progress_gain = (focus_score / 100) * config['reward_sensitivity']
        elif action == 'take_break':
            progress_gain = 0
        else:
            progress_gain = -(focus_score / 100) * 0.5
        
        self.task_progress += progress_gain
        self.task_progress = max(0, min(self.task_progress, self.task_completion_threshold))
        
        return self.task_progress
    
    def is_completed(self) -> bool:
        return self.task_progress >= self.task_completion_threshold
    
    def grade_trajectory(self, trajectory: List[Dict]) -> float:
        if not trajectory:
            return 0.0
        
        total_reward = sum(t.get('reward', 0) for t in trajectory)
        avg_focus = np.mean([t.get('focus_score', 0) for t in trajectory])
        
        score = (avg_focus / 100) * 0.6 + (total_reward / len(trajectory) / 20) * 0.4
        return min(1.0, max(0.0, score))
    
    def reset(self):
        self.task_progress = 0