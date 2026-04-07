import asyncio
import re
from typing import Dict, List, Optional

class ScreenIntelligence:
    def __init__(self):
        self.distraction_keywords = [
            'youtube', 'netflix', 'facebook', 'instagram', 'twitter',
            'game', 'play', 'watch', 'movie', 'entertainment', 'tiktok',
            'reddit', 'discord', 'twitch', 'spotify', 'netflix'
        ]
        
        self.study_keywords = [
            'learn', 'course', 'tutorial', 'documentation', 'lecture',
            'research', 'study', 'assignment', 'homework', 'class',
            'python', 'javascript', 'coding', 'algorithm', 'datascience',
            'machine learning', 'ai', 'coursera', 'udemy', 'edx'
        ]
    
    async def extract_screen_text(self, screenshot_bytes: bytes) -> str:
        """Extract text from screen using OCR (simplified for demo)"""
        # In production, use pytesseract
        # For demo, return sample text
        return "Studying machine learning algorithms and neural networks"
    
    async def classify_content(self, text: str) -> Dict:
        """Classify content as study or distraction"""
        if not text.strip():
            return {"label": "neutral", "score": 0.5}
        
        text_lower = text.lower()
        distraction_score = sum(1 for kw in self.distraction_keywords if kw in text_lower)
        study_score = sum(1 for kw in self.study_keywords if kw in text_lower)
        
        total = distraction_score + study_score
        if total == 0:
            return {"label": "neutral", "score": 0.5}
        
        study_ratio = study_score / total
        
        final_score = study_ratio
        
        return {
            "label": "study" if final_score > 0.6 else "distraction" if final_score < 0.4 else "neutral",
            "score": final_score,
            "study_keywords": study_score,
            "distraction_keywords": distraction_score
        }
    
    async def analyze_activity(self, screenshot_bytes: Optional[bytes] = None) -> Dict:
        """Complete screen analysis pipeline"""
        if screenshot_bytes:
            text = await self.extract_screen_text(screenshot_bytes)
        else:
            text = "Productive study session in progress"
        
        classification = await self.classify_content(text)
        
        return {
            "text": text[:200],
            "classification": classification,
            "productivity_score": classification['score'],
            "timestamp": None
        }