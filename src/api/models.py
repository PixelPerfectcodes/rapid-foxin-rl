from pydantic import BaseModel
from typing import Optional, List

class ActionRequest(BaseModel):
    action: int
    ai_focus_score: Optional[float] = None

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"
    random_seed: Optional[int] = None

class StepResponse(BaseModel):
    status: str
    state: dict
    reward: float
    done: bool
    loss: float

class MetricsResponse(BaseModel):
    status: str
    episodes: int
    avg_reward: float
    avg_loss: float
    epsilon: float
    memory_size: int
    focus_streak: int
    fatigue_level: float