from typing import Dict, Optional
from datetime import datetime
import numpy as np

class FusionEngine:
    def __init__(self):
        self.history = []
        self.history_length = 10
    
    async def fuse_signals(self, screen_data: Dict, camera_data: Dict, behavioral_data: Optional[Dict] = None) -> Dict:
        """Fuse multimodal signals into unified focus score"""
        
        # Extract scores
        screen_score = screen_data.get('productivity_score', 0.5) * 100
        
        # Get camera analysis
        camera_score = camera_data.get('focus_confidence', 50)
        face_detected = camera_data.get('face_detected', False)
        text_detected = camera_data.get('text_detected', False)
        text_classification = camera_data.get('text_classification', {})
        attention_score = camera_data.get('attention_score', 0.5)
        
        # Dynamic weights based on available data
        if face_detected and text_detected:
            weights = {'screen': 0.20, 'camera_face': 0.40, 'camera_text': 0.30, 'behavioral': 0.10}
        elif face_detected:
            weights = {'screen': 0.30, 'camera_face': 0.50, 'camera_text': 0.10, 'behavioral': 0.10}
        elif text_detected:
            weights = {'screen': 0.35, 'camera_face': 0.15, 'camera_text': 0.40, 'behavioral': 0.10}
        else:
            weights = {'screen': 0.60, 'camera_face': 0.20, 'camera_text': 0.10, 'behavioral': 0.10}
        
        # Calculate base focus
        base_focus = screen_score * weights['screen']
        base_focus += camera_score * weights['camera_face']
        
        # Add text classification bonus/penalty
        if text_classification:
            if text_classification.get('label') == 'study':
                text_bonus = 20 * text_classification.get('confidence', 0.5)
                base_focus += text_bonus * weights['camera_text']
            elif text_classification.get('label') == 'distraction':
                text_penalty = 25 * text_classification.get('confidence', 0.5)
                base_focus -= text_penalty * weights['camera_text']
        
        # Add attention score influence
        base_focus += attention_score * 20 * weights['camera_face']
        
        # Add behavioral adjustments
        if behavioral_data:
            streak_bonus = behavioral_data.get('focus_streak', 0) * 0.5
            fatigue_penalty = behavioral_data.get('fatigue_level', 0) * 30
            base_focus += streak_bonus - fatigue_penalty
        
        # Temporal smoothing
        focus_score = self._temporal_smoothing(base_focus)
        
        # Determine attention state
        if focus_score >= 85:
            attention_state = "deep_focus"
        elif focus_score >= 70:
            attention_state = "focused"
        elif focus_score >= 50:
            attention_state = "neutral"
        elif focus_score >= 30:
            attention_state = "distracted"
        else:
            attention_state = "tired"
        
        # Calculate confidence based on data quality
        confidence = 0.5
        if face_detected:
            confidence += 0.25
        if text_detected and text_classification.get('confidence', 0) > 0.5:
            confidence += 0.2
        if behavioral_data:
            confidence += 0.05
        
        result = {
            "focus_score": round(max(0, min(100, focus_score)), 2),
            "attention_state": attention_state,
            "confidence": min(0.95, confidence),
            "components": {
                "screen": round(screen_score, 2),
                "camera_face": round(camera_score, 2),
                "face_detected": face_detected,
                "text_detected": text_detected,
                "text_activity": text_classification.get('label', 'unknown'),
                "attention_score": round(attention_score, 3)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(result)
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        return result
    
    def _temporal_smoothing(self, current_score: float) -> float:
        if not self.history:
            return current_score
        
        alpha = 0.7
        last_score = self.history[-1]['focus_score']
        return alpha * current_score + (1 - alpha) * last_score
    
    def get_prediction(self) -> Dict:
        if len(self.history) < 3:
            return {"prediction": "insufficient_data", "confidence": 0.0}
        
        recent_scores = [h['focus_score'] for h in self.history[-5:]]
        
        if len(recent_scores) > 1:
            # Calculate trend using linear regression
            x = np.arange(len(recent_scores))
            z = np.polyfit(x, recent_scores, 1)
            trend = z[0]
        else:
            trend = 0
        
        if trend > 2:
            prediction = "improving"
        elif trend < -2:
            prediction = "declining"
        else:
            prediction = "stable"
        
        # Predict next state
        next_score = recent_scores[-1] + trend
        if next_score >= 85:
            next_state = "deep_focus"
        elif next_score >= 70:
            next_state = "focused"
        elif next_score >= 50:
            next_state = "neutral"
        elif next_score >= 30:
            next_state = "distracted"
        else:
            next_state = "tired"
        
        return {
            "prediction": prediction,
            "next_state": next_state,
            "trend": round(trend, 2),
            "confidence": min(0.9, abs(trend) / 20)
        }