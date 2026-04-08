#!/usr/bin/env python3
"""
Rapid Foxin - AI Student Productivity System
Inference script for production deployment with TrOCR for OCR

Follows strict requirements:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- OpenAI client configuration
- Structured stdout logging (START/STEP/END)
- TrOCR for image-to-text extraction
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional, Any, List
from io import BytesIO
from PIL import Image

import requests
import numpy as np
from openai import OpenAI

# Import transformers for TrOCR
from transformers import (
    AutoTokenizer, 
    AutoModelForVision2Seq,
    pipeline
)
import torch

import cv2

# Configure logging for structured output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format for structured logs
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class RapidFoxinInference:
    """
    Inference class for Rapid Foxin AI Student Productivity System
    Uses TrOCR for OCR and OpenAI client for action prediction
    """
    
    def __init__(self):
        """Initialize inference engine with environment variables"""
        # Environment variables (required)
        self.api_base_url = os.getenv("API_BASE_URL")
        self.model_name = os.getenv("MODEL_NAME")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Optional
        self.local_image_name = os.getenv("LOCAL_IMAGE_NAME")
        
        # Validate required environment variables
        self._validate_env_vars()
        
        # Set Hugging Face token for transformers
        os.environ["HF_TOKEN"] = self.hf_token
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.hf_token  # HF_TOKEN used as API key
        )
        
        # Initialize TrOCR for OCR
        self.ocr_model_name = "DunnBC22/trocr-large-printed-cmc7_tesseract_MICR_ocr"
        self.ocr_tokenizer = None
        self.ocr_model = None
        self.ocr_pipeline = None
        self._init_ocr_model()
        
        # System configuration
        self.state_names = ['focused', 'distracted', 'tired', 'deep_focus']
        self.action_names = ['study', 'take_break', 'use_phone', 'switch_task']
        
        # Study vs distraction keywords
        self.study_keywords = [
            'python', 'java', 'javascript', 'code', 'coding', 'programming',
            'algorithm', 'data', 'machine learning', 'ai', 'deep learning',
            'tutorial', 'course', 'lecture', 'class', 'study', 'learn',
            'homework', 'assignment', 'exam', 'test', 'quiz', 'research',
            'book', 'textbook', 'documentation', 'api', 'framework',
            'equation', 'formula', 'theorem', 'proof', 'analysis'
        ]
        
        self.distraction_keywords = [
            'youtube', 'netflix', 'facebook', 'instagram', 'twitter',
            'tiktok', 'snapchat', 'reddit', 'discord', 'twitch',
            'game', 'gaming', 'play', 'watch', 'movie', 'show',
            'sports', 'entertainment', 'memes', 'funny', 'viral',
            'stream', 'video', 'music', 'spotify', 'netflix'
        ]
        
        # State tracking
        self.current_state = None
        self.focus_history = []
        self.reward_history = []
        self.ocr_history = []
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Camera setup
        self.camera = None
        self.camera_enabled = False
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info("START: Rapid Foxin Inference Engine Initialized")
        logger.info(f"STEP: API_BASE_URL configured: {self.api_base_url if self.api_base_url else 'Not set'}")
        logger.info(f"STEP: MODEL_NAME: {self.model_name}")
        logger.info(f"STEP: HF_TOKEN: {'✓ Present' if self.hf_token else '✗ Missing'}")
        logger.info(f"STEP: TrOCR Model: {self.ocr_model_name}")
        logger.info(f"STEP: Local image: {self.local_image_name if self.local_image_name else 'Not using Docker image'}")
    
    def _validate_env_vars(self):
        """Validate required environment variables"""
        missing = []
        if not self.api_base_url:
            missing.append("API_BASE_URL")
        if not self.model_name:
            missing.append("MODEL_NAME")
        if not self.hf_token:
            missing.append("HF_TOKEN")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set API_BASE_URL, MODEL_NAME, and HF_TOKEN"
            )
    
    def _init_ocr_model(self):
        """Initialize TrOCR model for OCR"""
        try:
            logger.info("STEP: Loading TrOCR model for OCR...")
            
            # Load tokenizer and model
            self.ocr_tokenizer = AutoTokenizer.from_pretrained(
                self.ocr_model_name,
                token=self.hf_token
            )
            
            self.ocr_model = AutoModelForVision2Seq.from_pretrained(
                self.ocr_model_name,
                token=self.hf_token
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.ocr_model = self.ocr_model.cuda()
                logger.info("STEP: TrOCR model loaded on GPU")
            else:
                logger.info("STEP: TrOCR model loaded on CPU")
            
            # Create pipeline
            self.ocr_pipeline = pipeline(
                "image-to-text",
                model=self.ocr_model,
                tokenizer=self.ocr_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("STEP: TrOCR model initialized successfully")
            
        except Exception as e:
            logger.info(f"STEP: TrOCR initialization failed: {e}")
            logger.info("STEP: Falling back to Tesseract OCR")
            self.ocr_pipeline = None
    
    def log_structured(self, event: str, data: Dict = None):
        """Log structured output with START/STEP/END format"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {}
        }
        logger.info(json.dumps(log_entry))
    
    def extract_text_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text from image using TrOCR or fallback
        Returns extracted text and classification
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            extracted_text = ""
            confidence = 0.0
            
            # Use TrOCR if available
            if self.ocr_pipeline is not None:
                result = self.ocr_pipeline(pil_image)
                if result and len(result) > 0:
                    extracted_text = result[0].get('generated_text', '')
                    confidence = 0.85  # Default confidence for TrOCR
            
            # Fallback: simple text detection if TrOCR fails
            if not extracted_text or len(extracted_text) < 5:
                # Use basic text detection (placeholder)
                extracted_text = self._simple_text_detection(image)
                confidence = 0.5
            
            # Classify the extracted text
            classification = self._classify_text_content(extracted_text)
            
            # Store in history
            self.ocr_history.append({
                "text": extracted_text[:200],
                "classification": classification,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history limited
            if len(self.ocr_history) > 50:
                self.ocr_history.pop(0)
            
            return {
                "extracted_text": extracted_text,
                "confidence": confidence,
                "classification": classification,
                "text_length": len(extracted_text)
            }
            
        except Exception as e:
            self.log_structured("STEP", {"event": "ocr_error", "error": str(e)})
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "classification": {"label": "neutral", "score": 0.5},
                "text_length": 0
            }
    
    def _simple_text_detection(self, image: np.ndarray) -> str:
        """Simple text detection fallback using image moments"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find text regions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (potential text regions)
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 10 < w < 500 and 5 < h < 100:  # Typical text dimensions
                    text_regions.append((x, y, w, h))
            
            if text_regions:
                return f"Detected {len(text_regions)} text regions"
            return ""
            
        except Exception:
            return ""
    
    def _classify_text_content(self, text: str) -> Dict:
        """Classify text as study-related or distraction"""
        if not text or len(text) < 5:
            return {"label": "neutral", "score": 0.5, "confidence": 0.2}
        
        text_lower = text.lower()
        
        # Count matching keywords
        study_matches = [kw for kw in self.study_keywords if kw in text_lower]
        distraction_matches = [kw for kw in self.distraction_keywords if kw in text_lower]
        
        study_score = len(study_matches)
        distraction_score = len(distraction_matches)
        
        total_matches = study_score + distraction_score
        
        if total_matches > 0:
            study_ratio = study_score / total_matches
            confidence = min(0.9, total_matches / 15)
        else:
            study_ratio = 0.5
            confidence = 0.15
        
        if study_ratio > 0.6:
            label = "study"
            score = study_ratio
        elif study_ratio < 0.4:
            label = "distraction"
            score = 1 - study_ratio
        else:
            label = "neutral"
            score = 0.5
        
        return {
            "label": label,
            "score": float(score),
            "confidence": float(confidence),
            "study_matches": study_score,
            "distraction_matches": distraction_score,
            "matched_keywords": study_matches[:5] + distraction_matches[:5]
        }
    
    def get_state_vector(self, state: Dict) -> np.ndarray:
        """Convert state dict to vector for model input"""
        vector = np.zeros(len(self.state_names) + 3)
        
        # One-hot encode current state
        if state['current_state'] in self.state_names:
            idx = self.state_names.index(state['current_state'])
            vector[idx] = 1
        
        # Add continuous features
        vector[-3] = state.get('fatigue', 0.0)
        vector[-2] = min(1.0, state.get('focus_streak', 0) / 100)
        vector[-1] = state.get('attention_drift', 0.0)
        
        return vector
    
    def predict_action(self, state_vector: np.ndarray, ocr_classification: Dict = None) -> int:
        """
        Predict best action using LLM via OpenAI client
        Returns action index (0-3)
        """
        try:
            # Prepare prompt for LLM
            state_description = self._state_vector_to_text(state_vector)
            
            # Add OCR context if available
            ocr_context = ""
            if ocr_classification and ocr_classification.get('label') != 'neutral':
                ocr_context = f"\nScreen Content Analysis: {ocr_classification['label'].upper()} activity detected (confidence: {ocr_classification.get('confidence', 0):.2f})"
                
                if ocr_classification.get('study_matches', 0) > 0:
                    ocr_context += f"\n- Study-related keywords detected: {ocr_classification.get('study_matches', 0)} matches"
                if ocr_classification.get('distraction_matches', 0) > 0:
                    ocr_context += f"\n- Distraction-related keywords detected: {ocr_classification.get('distraction_matches', 0)} matches"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI productivity coach. Based on the student's current state and screen content, recommend the best action to maximize focus and learning outcomes.
                        
Available actions:
0: study - Continue working on academic tasks
1: take_break - Rest to reduce fatigue
2: use_phone - Check phone (usually leads to distraction)
3: switch_task - Change to a different subject/task

Return ONLY the action number (0-3) as a single digit."""
                    },
                    {
                        "role": "user",
                        "content": f"Current student state:{state_description}{ocr_context}\n\nRecommended action:"
                    }
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            # Parse response
            action_text = response.choices[0].message.content.strip()
            action = int(action_text[0]) if action_text else 0
            action = max(0, min(3, action))  # Clamp to valid range
            
            self.log_structured("STEP", {"event": "action_prediction", "action": action, "action_name": self.action_names[action]})
            return action
            
        except Exception as e:
            self.log_structured("STEP", {"event": "prediction_error", "error": str(e)})
            # Fallback to rule-based decision
            return self._fallback_action(state_vector, ocr_classification)
    
    def _state_vector_to_text(self, state_vector: np.ndarray) -> str:
        """Convert state vector to human-readable description"""
        # Find current state
        state_idx = np.argmax(state_vector[:4]) if np.max(state_vector[:4]) > 0 else 0
        current_state = self.state_names[state_idx]
        
        fatigue = state_vector[-3]
        focus_streak = int(state_vector[-2] * 100)
        attention_drift = state_vector[-1]
        
        description = f"""
- Current State: {current_state}
- Fatigue Level: {fatigue:.2f} (0=energized, 1=exhausted)
- Focus Streak: {focus_streak} minutes of continuous focus
- Attention Drift: {attention_drift:.2f} (0=stable, 1=highly distracted)

Student is {'highly focused' if focus_streak > 50 else 'moderately focused' if focus_streak > 20 else 'struggling with focus'}.
{'Student appears tired and may need a break.' if fatigue > 0.7 else ''}
{'Student is in a good flow state.' if focus_streak > 30 and fatigue < 0.5 else ''}"""
        
        return description
    
    def _fallback_action(self, state_vector: np.ndarray, ocr_classification: Dict = None) -> int:
        """Rule-based fallback when LLM is unavailable"""
        fatigue = state_vector[-3]
        focus_streak = state_vector[-2] * 100
        
        # Check OCR classification
        if ocr_classification:
            if ocr_classification.get('label') == 'distraction':
                return 0  # study (to get back on track)
            elif ocr_classification.get('label') == 'study':
                if fatigue > 0.6:
                    return 1  # take_break
                return 0  # study
        
        if fatigue > 0.7:
            return 1  # take_break
        elif focus_streak > 50:
            return 0  # study
        elif fatigue > 0.4:
            return 1  # take_break
        else:
            return 0  # study
    
    def compute_reward(self, state: Dict, action: int, ocr_classification: Dict = None) -> float:
        """Compute reward based on state transition and action"""
        reward = 0.0
        
        # Base reward based on state
        state_rewards = {
            'deep_focus': 20,
            'focused': 10,
            'neutral': 0,
            'distracted': -10,
            'tired': -5
        }
        
        state_name = state.get('current_state', 'neutral')
        reward += state_rewards.get(state_name, 0)
        
        # OCR-based bonus/penalty
        if ocr_classification:
            if ocr_classification.get('label') == 'study':
                reward += 5 * ocr_classification.get('confidence', 0.5)
            elif ocr_classification.get('label') == 'distraction':
                reward -= 10 * ocr_classification.get('confidence', 0.5)
        
        # Fatigue penalty
        reward -= state.get('fatigue', 0) * 15
        
        # Focus streak bonus
        reward += min(10, state.get('focus_streak', 0) / 10)
        
        # Action-specific adjustments
        if action == 0 and state_name in ['focused', 'deep_focus']:
            reward += 5
        elif action == 1 and state_name == 'tired':
            reward += 3
        elif action == 2:
            reward -= 8
        
        return reward
    
    def analyze_camera_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze camera frame for face detection and OCR text extraction
        Returns focus score, attention metrics, and extracted text
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            face_analysis = {
                "face_detected": len(faces) > 0,
                "face_count": len(faces),
                "attention_score": 50.0,
                "focus_score": 50.0
            }
            
            if len(faces) > 0:
                # Calculate attention score based on face position and size
                x, y, w, h = faces[0]
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                
                # Position score (centered is better)
                position_score_x = 1 - min(1, abs(face_center_x - frame_center_x) / frame_center_x)
                position_score_y = 1 - min(1, abs(face_center_y - frame_center_y) / frame_center_y)
                position_score = (position_score_x + position_score_y) / 2
                
                # Size score (closer is better)
                size_score = min(1, (w * h) / (frame.shape[0] * frame.shape[1]) / 0.15)
                
                attention_score = (position_score * 0.4 + size_score * 0.6) * 100
                
                face_analysis["attention_score"] = attention_score
                face_analysis["focus_score"] = attention_score
                face_analysis["faces"] = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}]
                face_analysis["confidence"] = 0.8
            else:
                face_analysis["confidence"] = 0.3
            
            # Extract text using TrOCR
            ocr_result = self.extract_text_from_image(frame)
            
            # Combine results
            result = {
                **face_analysis,
                "ocr": ocr_result,
                "overall_focus_score": (
                    face_analysis["focus_score"] * 0.6 + 
                    (ocr_result["classification"]["score"] * 100) * 0.4
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.log_structured("STEP", {"event": "camera_error", "error": str(e)})
            return {
                "face_detected": False,
                "face_count": 0,
                "attention_score": 50.0,
                "focus_score": 50.0,
                "confidence": 0.5,
                "ocr": {"extracted_text": "", "classification": {"label": "neutral", "score": 0.5}},
                "overall_focus_score": 50.0
            }
    
    def run_inference(self, state: Dict, camera_frame: Optional[np.ndarray] = None) -> Dict:
        """
        Main inference loop
        Takes current state and optional camera frame, returns action and metrics
        """
        self.log_structured("STEP", {"event": "inference_start", "state": state.get('current_state', 'unknown')})
        
        # Get state vector
        state_vector = self.get_state_vector(state)
        
        # Analyze camera if available
        camera_analysis = None
        ocr_classification = None
        
        if camera_frame is not None:
            camera_analysis = self.analyze_camera_frame(camera_frame)
            ocr_classification = camera_analysis.get('ocr', {}).get('classification')
            
            # Update state with camera insights
            if camera_analysis.get('face_detected'):
                attention = camera_analysis.get('attention_score', 50)
                if attention < 40:
                    state['attention_drift'] = min(1.0, state.get('attention_drift', 0) + 0.1)
                elif attention > 70:
                    state['attention_drift'] = max(0, state.get('attention_drift', 0) - 0.05)
            
            # Update focus score based on OCR
            if ocr_classification:
                if ocr_classification.get('label') == 'study':
                    state['focus_score'] = min(100, state.get('focus_score', 50) + 5)
                elif ocr_classification.get('label') == 'distraction':
                    state['focus_score'] = max(0, state.get('focus_score', 50) - 10)
        
        # Predict best action with OCR context
        action = self.predict_action(state_vector, ocr_classification)
        
        # Compute expected reward
        expected_reward = self.compute_reward(state, action, ocr_classification)
        
        # Update history
        self.focus_history.append(state.get('focus_score', 50))
        self.reward_history.append(expected_reward)
        
        # Keep history limited
        if len(self.focus_history) > 100:
            self.focus_history.pop(0)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        result = {
            "action": action,
            "action_name": self.action_names[action],
            "expected_reward": expected_reward,
            "state_vector": state_vector.tolist(),
            "current_state": state.get('current_state', 'unknown'),
            "focus_score": state.get('focus_score', 50),
            "fatigue": state.get('fatigue', 0),
            "focus_streak": state.get('focus_streak', 0),
            "camera_analysis": camera_analysis,
            "ocr_text": camera_analysis.get('ocr', {}).get('extracted_text', '')[:200] if camera_analysis else None,
            "ocr_classification": ocr_classification,
            "confidence": camera_analysis.get('confidence', 0.7) if camera_analysis else 0.6
        }
        
        self.log_structured("STEP", {"event": "inference_complete", "result": {
            "action": result["action_name"],
            "expected_reward": expected_reward,
            "focus_score": result["focus_score"],
            "ocr_label": ocr_classification.get('label') if ocr_classification else None
        }})
        
        return result
    
    def train_step(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Update model based on experience"""
        self.total_reward += reward
        
        if done:
            self.episode_count += 1
            avg_reward = self.total_reward / self.episode_count if self.episode_count > 0 else 0
            
            self.log_structured("STEP", {
                "event": "episode_complete",
                "episode": self.episode_count,
                "total_reward": self.total_reward,
                "avg_reward": avg_reward
            })
            
            # Send training feedback
            feedback = {
                "episode": self.episode_count,
                "total_reward": self.total_reward,
                "avg_reward": avg_reward,
                "final_state": next_state.get('current_state', 'unknown')
            }
            
            self.log_structured("STEP", {"event": "training_feedback", "data": feedback})
            
            # Reset for next episode
            self.total_reward = 0
    
    def capture_camera_frame(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """Capture a single frame from webcam"""
        try:
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(camera_id)
                self.camera_enabled = True
            
            ret, frame = self.camera.read()
            if ret:
                return frame
            return None
            
        except Exception as e:
            self.log_structured("STEP", {"event": "camera_capture_error", "error": str(e)})
            return None
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.camera_enabled = False
    
    def health_check(self) -> Dict:
        """Check system health and API connectivity"""
        health = {
            "status": "healthy",
            "api_base_url": self.api_base_url,
            "model_name": self.model_name,
            "hf_token_present": bool(self.hf_token),
            "trocr_loaded": self.ocr_pipeline is not None,
            "camera_enabled": self.camera_enabled,
            "episodes": self.episode_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test API connectivity
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0
            )
            health["api_connected"] = True
            health["api_response"] = response.choices[0].message.content[:50]
        except Exception as e:
            health["api_connected"] = False
            health["api_error"] = str(e)
        
        return health


def main():
    """Main inference loop for Rapid Foxin"""
    parser = argparse.ArgumentParser(description="Rapid Foxin AI Inference with TrOCR")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "continuous", "test"],
                        help="Inference mode: single prediction, continuous loop, or test")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for continuous mode")
    parser.add_argument("--camera", action="store_true",
                        help="Enable camera for multimodal perception")
    parser.add_argument("--health", action="store_true",
                        help="Run health check only")
    parser.add_argument("--image", type=str,
                        help="Path to image file for OCR testing")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        engine = RapidFoxinInference()
        
        # Health check mode
        if args.health:
            health = engine.health_check()
            engine.log_structured("END", {"event": "health_check", "health": health})
            print(json.dumps(health, indent=2))
            return
        
        # Test OCR on image if provided
        if args.image:
            engine.log_structured("STEP", {"event": "ocr_test", "image": args.image})
            image = cv2.imread(args.image)
            if image is not None:
                result = engine.extract_text_from_image(image)
                print("\n" + "="*50)
                print("OCR EXTRACTION RESULT")
                print("="*50)
                print(f"Extracted Text: {result['extracted_text'][:500]}")
                print(f"Classification: {result['classification']['label']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Study Matches: {result['classification'].get('study_matches', 0)}")
                print(f"Distraction Matches: {result['classification'].get('distraction_matches', 0)}")
                print("="*50)
            else:
                print(f"Error: Could not load image from {args.image}")
            return
        
        # Test mode
        if args.mode == "test":
            engine.log_structured("STEP", {"event": "test_mode", "message": "Running inference tests"})
            
            # Test state
            test_state = {
                "current_state": "focused",
                "fatigue": 0.3,
                "focus_streak": 25,
                "attention_drift": 0.1,
                "focus_score": 75
            }
            
            # Run inference
            result = engine.run_inference(test_state)
            engine.log_structured("STEP", {"event": "test_result", "result": result})
            
            # Test training step
            engine.train_step(
                state=test_state,
                action=result["action"],
                reward=result["expected_reward"],
                next_state={"current_state": "deep_focus"},
                done=False
            )
            
            engine.log_structured("END", {"event": "test_complete", "status": "success"})
        
        # Single prediction mode
        elif args.mode == "single":
            engine.log_structured("STEP", {"event": "single_mode", "message": "Running single prediction"})
            
            # Example state
            current_state = {
                "current_state": "focused",
                "fatigue": 0.2,
                "focus_streak": 15,
                "attention_drift": 0.05,
                "focus_score": 80
            }
            
            # Capture camera frame if enabled
            camera_frame = None
            if args.camera:
                camera_frame = engine.capture_camera_frame()
            
            # Run inference
            result = engine.run_inference(current_state, camera_frame)
            
            # Output result
            print("\n" + "="*50)
            print("RAPID FOXIN INFERENCE RESULT")
            print("="*50)
            print(f"Recommended Action: {result['action_name']}")
            print(f"Expected Reward: {result['expected_reward']:.2f}")
            print(f"Current State: {result['current_state']}")
            print(f"Focus Score: {result['focus_score']:.1f}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            if result.get('ocr_text'):
                print(f"\nOCR Extracted Text Preview: {result['ocr_text'][:200]}")
            if result.get('ocr_classification'):
                print(f"Screen Classification: {result['ocr_classification'].get('label', 'unknown')}")
            
            print("="*50)
            
            engine.log_structured("END", {"event": "single_prediction_complete", "result": result})
            
            # Release camera
            if args.camera:
                engine.release_camera()
        
        # Continuous mode
        elif args.mode == "continuous":
            engine.log_structured("STEP", {"event": "continuous_mode", "episodes": args.episodes})
            
            if args.camera:
                engine.log_structured("STEP", {"event": "camera_enabled"})
            
            for episode in range(args.episodes):
                engine.log_structured("STEP", {"event": "episode_start", "episode": episode + 1})
                
                state = {
                    "current_state": "focused",
                    "fatigue": max(0, min(1, episode * 0.05)),
                    "focus_streak": episode * 2,
                    "attention_drift": max(0, min(1, episode * 0.02)),
                    "focus_score": max(0, min(100, 80 - episode * 2))
                }
                
                camera_frame = None
                if args.camera:
                    camera_frame = engine.capture_camera_frame()
                
                result = engine.run_inference(state, camera_frame)
                
                next_state = {
                    "current_state": "deep_focus" if result["action"] == 0 else "distracted" if result["action"] == 2 else "focused",
                    "fatigue": state["fatigue"] + 0.05,
                    "focus_streak": state["focus_streak"] + 1 if result["action"] == 0 else max(0, state["focus_streak"] - 2),
                    "attention_drift": state["attention_drift"] + 0.02 if result["action"] == 2 else max(0, state["attention_drift"] - 0.01),
                    "focus_score": state["focus_score"] + result["expected_reward"] / 2
                }
                
                done = episode == args.episodes - 1 or next_state["fatigue"] > 0.9
                engine.train_step(state, result["action"], result["expected_reward"], next_state, done)
                
                time.sleep(1)
            
            engine.log_structured("END", {"event": "continuous_mode_complete", "total_episodes": args.episodes})
            
            if args.camera:
                engine.release_camera()
        
    except Exception as e:
        logger.info(json.dumps({"event": "error", "error": str(e)}))
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()