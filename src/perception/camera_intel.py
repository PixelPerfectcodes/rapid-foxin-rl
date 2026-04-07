import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import base64
from typing import Dict, Optional, List, Tuple
import asyncio
from datetime import datetime
import re
import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraIntelligence:
    def __init__(self):
        self.camera = None
        self.is_camera_active = False
        self.current_frame = None
        self.ocr_history = []
        self.text_buffer = []
        self.face_history = []
        self.expression_history = []
        self.last_ocr_time = None
        self.frame_count = 0
        
        # Configure Tesseract path
        import sys
        import platform
        
        if platform.system() == 'Windows':
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract found at: {path}")
                    break
        elif platform.system() == 'Darwin':
            pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        else:
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Load pre-trained emotion detection model files
        self.emotion_model_path = self._download_emotion_model()
        self.emotion_net = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self._load_emotion_model()
        
        # Facial Action Units mapping for detailed expression analysis
        self.facial_landmarks_indices = {
            'jaw': list(range(0, 17)),
            'eyebrow_left': list(range(17, 22)),
            'eyebrow_right': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'eye_left': list(range(36, 42)),
            'eye_right': list(range(42, 48)),
            'mouth_outer': list(range(48, 60)),
            'mouth_inner': list(range(60, 68))
        }
        
        # Keywords for classification
        self.study_keywords = [
            'python', 'java', 'javascript', 'code', 'coding', 'programming',
            'algorithm', 'data', 'machine learning', 'ai', 'deep learning',
            'tutorial', 'course', 'lecture', 'class', 'study', 'learn',
            'homework', 'assignment', 'exam', 'test', 'quiz', 'research',
            'book', 'textbook', 'documentation', 'api', 'framework'
        ]
        
        self.distraction_keywords = [
            'youtube', 'netflix', 'facebook', 'instagram', 'twitter',
            'tiktok', 'snapchat', 'reddit', 'discord', 'twitch',
            'game', 'gaming', 'play', 'watch', 'movie', 'show',
            'sports', 'entertainment', 'memes', 'funny', 'viral'
        ]
        
        logger.info("Camera Intelligence initialized with Facial Expression Detection")
    
    def _download_emotion_model(self):
        """Download pre-trained emotion detection model if not exists"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Using OpenCV's face detection for emotion (simplified)
        # In production, you'd download a proper emotion detection model
        # For now, we'll use a simplified approach with facial features
        
        return model_dir
    
    def _load_emotion_model(self):
        """Load emotion detection model"""
        try:
            # For production, use a proper emotion detection DNN model
            # Example: emotion_model = cv2.dnn.readNetFromCaffe(proto, model)
            # For now, we'll use a feature-based approach
            self.emotion_net = None
            logger.info("Emotion detection model loaded (feature-based)")
        except Exception as e:
            logger.warning(f"Could not load emotion model: {e}")
            self.emotion_net = None
    
    async def start_camera(self, camera_id: int = 0) -> bool:
        """Start the camera feed"""
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(camera_id)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    self.is_camera_active = True
                    logger.info("Camera started successfully")
                    return True
                else:
                    logger.error("Failed to open camera")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    async def stop_camera(self) -> bool:
        """Stop the camera feed"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.is_camera_active = False
            logger.info("Camera stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            return False
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        try:
            if self.camera is not None and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame
                    self.frame_count += 1
                    return frame
            return None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    async def detect_facial_expressions(self, face_roi: np.ndarray) -> Dict:
        """Detect facial expressions using facial features and landmarks"""
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
            
            # Detect smile
            smiles = self.smile_cascade.detectMultiScale(gray_face, 1.1, 15)
            
            # Calculate facial features
            face_height, face_width = face_roi.shape[:2]
            
            # Eye aspect ratio (blink detection)
            eye_openness = len(eyes) / 2.0 if len(eyes) > 0 else 0
            
            # Smile intensity
            smile_intensity = len(smiles) / 2.0 if len(smiles) > 0 else 0
            
            # Calculate mouth aspect ratio (for smile detection)
            # Using image moments for basic shape analysis
            moments = cv2.moments(gray_face)
            if moments['m00'] != 0:
                # Simple shape analysis
                hu_moments = cv2.HuMoments(moments).flatten()
            else:
                hu_moments = [0] * 7
            
            # Detect expression based on features
            expression_scores = {
                'Happy': 0,
                'Sad': 0,
                'Angry': 0,
                'Surprised': 0,
                'Neutral': 0,
                'Focused': 0,
                'Distracted': 0,
                'Tired': 0
            }
            
            # Happy detection (smile + open eyes)
            if smile_intensity > 0.3:
                expression_scores['Happy'] += 0.7
                expression_scores['Focused'] += 0.3
            
            # Surprised detection (wide eyes + open mouth)
            if eye_openness > 0.8:
                expression_scores['Surprised'] += 0.6
                expression_scores['Focused'] += 0.2
            
            # Tired detection (closed/squinted eyes)
            if eye_openness < 0.3:
                expression_scores['Tired'] += 0.8
            
            # Focused detection (attentive eyes, slight smile or neutral)
            if 0.4 < eye_openness < 0.7 and smile_intensity < 0.2:
                expression_scores['Focused'] += 0.6
                expression_scores['Neutral'] += 0.3
            
            # Distracted detection (looking away, not engaged)
            if eye_openness < 0.4 and smile_intensity < 0.1:
                expression_scores['Distracted'] += 0.5
                expression_scores['Tired'] += 0.3
            
            # Sad detection (downturned mouth, less eye openness)
            if eye_openness < 0.5 and smile_intensity < 0.05:
                expression_scores['Sad'] += 0.5
            
            # Angry detection (furrowed brows - approximated)
            if eye_openness < 0.6 and smile_intensity < 0.1:
                expression_scores['Angry'] += 0.4
            
            # Get primary expression
            primary_expression = max(expression_scores, key=expression_scores.get)
            confidence = expression_scores[primary_expression]
            
            # Calculate emotion intensity
            emotion_intensity = confidence
            
            # Calculate engagement score
            engagement_score = (
                (expression_scores['Focused'] * 0.4) +
                (expression_scores['Happy'] * 0.3) +
                (expression_scores['Surprised'] * 0.2) -
                (expression_scores['Distracted'] * 0.3) -
                (expression_scores['Tired'] * 0.4) -
                (expression_scores['Sad'] * 0.2) -
                (expression_scores['Angry'] * 0.2)
            )
            engagement_score = max(0, min(1, engagement_score + 0.5))
            
            return {
                "primary_expression": primary_expression,
                "confidence": round(confidence, 3),
                "intensity": round(emotion_intensity, 3),
                "engagement_score": round(engagement_score, 3),
                "all_scores": {k: round(v, 3) for k, v in expression_scores.items()},
                "eye_openness": round(eye_openness, 3),
                "smile_intensity": round(smile_intensity, 3),
                "eyes_detected": len(eyes),
                "smile_detected": len(smiles) > 0
            }
            
        except Exception as e:
            logger.error(f"Expression detection error: {e}")
            return {
                "primary_expression": "Unknown",
                "confidence": 0,
                "intensity": 0,
                "engagement_score": 0.5,
                "all_scores": {},
                "eye_openness": 0.5,
                "smile_intensity": 0,
                "eyes_detected": 0,
                "smile_detected": False
            }
    
    async def detect_faces_with_expressions(self, frame: np.ndarray) -> Dict:
        """Detect faces and analyze facial expressions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with different scales for better detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60)
            )
            
            face_data = []
            total_attention = 0
            total_engagement = 0
            expressions = []
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect facial expressions
                expression_result = await self.detect_facial_expressions(face_roi)
                
                # Calculate face attention score
                face_center_x = x + w/2
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                face_center_y = y + h/2
                
                # Position score (centered is better)
                position_score_x = 1 - min(1, abs(face_center_x - frame_center_x) / (frame.shape[1] / 2))
                position_score_y = 1 - min(1, abs(face_center_y - frame_center_y) / (frame.shape[0] / 2))
                position_score = (position_score_x + position_score_y) / 2
                
                # Size score (closer is better)
                face_area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                size_score = min(1, face_area_ratio / 0.15)
                
                # Combined attention score
                attention_score = (
                    position_score * 0.25 +
                    size_score * 0.25 +
                    expression_result['engagement_score'] * 0.5
                )
                
                # Map expression to attention level
                expression_attention_map = {
                    'Focused': 0.9,
                    'Happy': 0.8,
                    'Surprised': 0.7,
                    'Neutral': 0.6,
                    'Sad': 0.4,
                    'Distracted': 0.3,
                    'Tired': 0.2,
                    'Angry': 0.3,
                    'Unknown': 0.5
                }
                
                expression_attention = expression_attention_map.get(
                    expression_result['primary_expression'], 0.5
                )
                
                face_info = {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "attention_score": float(attention_score),
                    "expression": expression_result['primary_expression'],
                    "expression_confidence": expression_result['confidence'],
                    "emotion_intensity": expression_result['intensity'],
                    "engagement_score": expression_result['engagement_score'],
                    "eye_openness": expression_result['eye_openness'],
                    "smile_intensity": expression_result['smile_intensity'],
                    "eyes_detected": expression_result['eyes_detected'],
                    "smile_detected": expression_result['smile_detected'],
                    "position_score": float(position_score),
                    "size_score": float(size_score)
                }
                
                face_data.append(face_info)
                total_attention += attention_score
                total_engagement += expression_result['engagement_score']
                expressions.append(expression_result['primary_expression'])
            
            # Calculate averages
            face_count = len(faces)
            avg_attention = total_attention / face_count if face_count > 0 else 0
            avg_engagement = total_engagement / face_count if face_count > 0 else 0
            
            # Determine overall expression
            if expressions:
                from collections import Counter
                overall_expression = Counter(expressions).most_common(1)[0][0]
            else:
                overall_expression = "No Face"
            
            return {
                "face_detected": face_count > 0,
                "face_count": face_count,
                "faces": face_data,
                "avg_attention_score": float(avg_attention),
                "avg_engagement_score": float(avg_engagement),
                "overall_expression": overall_expression,
                "expressions_detected": list(set(expressions))
            }
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {
                "face_detected": False,
                "face_count": 0,
                "faces": [],
                "avg_attention_score": 0.0,
                "avg_engagement_score": 0.0,
                "overall_expression": "Error",
                "expressions_detected": []
            }
    
    async def extract_text_from_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """Extract text from camera frame using Tesseract OCR"""
        try:
            # Preprocess frame for OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, h=30)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(denoised)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            
            # Extract text with confidence
            data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(conf))
            
            extracted_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return extracted_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return "", 0.0
    
    async def classify_text_content(self, text: str) -> Dict:
        """Classify extracted text as study or distraction"""
        if not text or len(text) < 5:
            return {"label": "neutral", "score": 0.5, "confidence": 0.2}
        
        text_lower = text.lower()
        
        # Find matching keywords
        study_matches = [kw for kw in self.study_keywords if kw in text_lower]
        distraction_matches = [kw for kw in self.distraction_keywords if kw in text_lower]
        
        study_score = len(study_matches)
        distraction_score = len(distraction_matches)
        
        total_matches = study_score + distraction_score
        
        if total_matches > 0:
            study_ratio = study_score / total_matches
            confidence = min(0.95, total_matches / 15)
        else:
            # Check for educational patterns
            educational_patterns = [
                r'\b(class|lecture|lesson|module)\b',
                r'\b(chapter|section|unit)\b',
                r'\b(learn|study|practice|exercise)\b',
                r'\b(quiz|test|exam|assessment)\b'
            ]
            
            for pattern in educational_patterns:
                if re.search(pattern, text_lower):
                    study_score += 1
            
            total_matches = study_score + distraction_score
            if total_matches > 0:
                study_ratio = study_score / total_matches
                confidence = min(0.7, total_matches / 10)
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
            "matched_keywords": study_matches[:5] + distraction_matches[:5],
            "text_preview": text[:300]
        }
    
    async def draw_expression_annotations(self, frame: np.ndarray, face_results: Dict, text_classification: Dict) -> np.ndarray:
        """Draw face and expression annotations on frame"""
        annotated = frame.copy()
        
        for face in face_results.get('faces', []):
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            
            # Color based on attention score
            if face['attention_score'] > 0.7:
                color = (0, 255, 0)  # Green - high attention
            elif face['attention_score'] > 0.4:
                color = (0, 255, 255)  # Yellow - medium attention
            else:
                color = (0, 0, 255)  # Red - low attention
            
            # Draw face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Draw expression and score
            expression_text = f"{face['expression']} ({face['expression_confidence']:.2f})"
            cv2.putText(annotated, expression_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw attention score
            score_text = f"Attention: {face['attention_score']:.2f}"
            cv2.putText(annotated, score_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw emotion emoji
            emoji_map = {
                'Happy': '😊',
                'Sad': '😢',
                'Angry': '😠',
                'Surprised': '😲',
                'Neutral': '😐',
                'Focused': '🎯',
                'Distracted': '😕',
                'Tired': '😴'
            }
            emoji = emoji_map.get(face['expression'], '❓')
            cv2.putText(annotated, emoji, (x+w-30, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw eye openness indicator
            eye_bar_width = int(face['eye_openness'] * 30)
            cv2.rectangle(annotated, (x, y+h+35), (x+eye_bar_width, y+h+40), (0, 255, 0), -1)
            
            # Draw smile intensity indicator
            smile_bar_width = int(face['smile_intensity'] * 30)
            cv2.rectangle(annotated, (x, y+h+45), (x+smile_bar_width, y+h+50), (0, 255, 255), -1)
        
        # Draw overall status
        if face_results['face_detected']:
            status_color = (0, 255, 0) if face_results['avg_attention_score'] > 0.6 else (0, 0, 255)
            status_text = f"Overall: {face_results['overall_expression']} | Engagement: {face_results['avg_engagement_score']:.2f}"
            cv2.putText(annotated, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw text classification
        if text_classification.get('label'):
            if text_classification['label'] == 'study':
                text_color = (0, 255, 0)
                label_text = "📚 STUDYING"
            elif text_classification['label'] == 'distraction':
                text_color = (0, 0, 255)
                label_text = "🎮 DISTRACTED"
            else:
                text_color = (255, 255, 0)
                label_text = "⚡ NEUTRAL"
            
            cv2.putText(annotated, label_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return annotated
    
    async def analyze_frame(self, frame_bytes: Optional[bytes] = None) -> Dict:
        """Complete frame analysis with face detection, expression recognition, and OCR"""
        try:
            # Get frame
            if frame_bytes:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = await self.get_frame()
            
            if frame is None:
                return self._default_response()
            
            # Detect faces with expressions
            face_results = await self.detect_faces_with_expressions(frame)
            
            # Extract text using OCR
            extracted_text, ocr_confidence = await self.extract_text_from_frame(frame)
            
            # Classify text
            text_classification = await self.classify_text_content(extracted_text)
            
            # Draw annotations
            annotated_frame = await self.draw_expression_annotations(frame, face_results, text_classification)
            
            # Calculate overall focus score
            focus_score = face_results['avg_attention_score'] * 100
            
            # Adjust based on text classification
            if text_classification['label'] == 'study':
                focus_score = min(100, focus_score + 15)
            elif text_classification['label'] == 'distraction':
                focus_score = max(0, focus_score - 25)
            
            # Adjust based on expression
            expression_bonus = {
                'Focused': 20,
                'Happy': 10,
                'Surprised': 5,
                'Neutral': 0,
                'Sad': -10,
                'Distracted': -20,
                'Tired': -15,
                'Angry': -15
            }
            focus_score += expression_bonus.get(face_results['overall_expression'], 0)
            focus_score = max(0, min(100, focus_score))
            
            # Store in history
            self.expression_history.append({
                "expression": face_results['overall_expression'],
                "attention_score": face_results['avg_attention_score'],
                "engagement_score": face_results['avg_engagement_score'],
                "timestamp": datetime.now()
            })
            
            if len(self.expression_history) > 30:
                self.expression_history.pop(0)
            
            # Convert annotated frame to base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "face_detected": face_results['face_detected'],
                "face_count": face_results['face_count'],
                "faces": face_results['faces'],
                "overall_expression": face_results['overall_expression'],
                "expressions_detected": face_results['expressions_detected'],
                "attention_score": round(face_results['avg_attention_score'], 3),
                "engagement_score": round(face_results['avg_engagement_score'], 3),
                "focus_score": round(focus_score, 1),
                "focus_confidence": round(focus_score, 1),
                "extracted_text": extracted_text[:500] if extracted_text else "",
                "text_classification": text_classification,
                "text_detected": len(extracted_text) > 10,
                "ocr_confidence": round(ocr_confidence, 1),
                "annotated_frame": annotated_frame_base64,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return self._default_response()
    
    async def analyze_frame_from_base64(self, base64_string: str) -> Dict:
        """Analyze camera frame from base64 string"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            frame_bytes = base64.b64decode(base64_string)
            return await self.analyze_frame(frame_bytes)
            
        except Exception as e:
            logger.error(f"Error analyzing base64 frame: {e}")
            return self._default_response()
    
    def _default_response(self) -> Dict:
        """Default response when analysis fails"""
        return {
            "face_detected": False,
            "face_count": 0,
            "faces": [],
            "overall_expression": "Unknown",
            "expressions_detected": [],
            "attention_score": 0.3,
            "engagement_score": 0.3,
            "focus_score": 30.0,
            "focus_confidence": 30.0,
            "extracted_text": "",
            "text_classification": {
                "label": "neutral",
                "score": 0.5,
                "confidence": 0.0
            },
            "text_detected": False,
            "ocr_confidence": 0.0,
            "annotated_frame": "",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_expression_summary(self) -> Dict:
        """Get summary of detected expressions over time"""
        if not self.expression_history:
            return {"has_data": False, "summary": "No expression data yet"}
        
        # Count expression frequencies
        from collections import Counter
        expressions = [e['expression'] for e in self.expression_history if e['expression'] != "No Face"]
        if not expressions:
            return {"has_data": False, "summary": "No face detected recently"}
        
        expression_counts = Counter(expressions)
        most_common = expression_counts.most_common(3)
        
        # Calculate average metrics
        avg_attention = np.mean([e['attention_score'] for e in self.expression_history])
        avg_engagement = np.mean([e['engagement_score'] for e in self.expression_history])
        
        # Determine overall mood
        positive_expressions = ['Happy', 'Focused', 'Surprised']
        negative_expressions = ['Sad', 'Angry', 'Distracted', 'Tired']
        
        positive_count = sum(expression_counts.get(e, 0) for e in positive_expressions)
        negative_count = sum(expression_counts.get(e, 0) for e in negative_expressions)
        
        if positive_count > negative_count:
            overall_mood = "positive"
        elif negative_count > positive_count:
            overall_mood = "negative"
        else:
            overall_mood = "neutral"
        
        return {
            "has_data": True,
            "total_frames": len(self.expression_history),
            "most_common_expressions": [{"expression": e, "count": c} for e, c in most_common],
            "avg_attention_score": round(avg_attention, 3),
            "avg_engagement_score": round(avg_engagement, 3),
            "overall_mood": overall_mood,
            "expression_distribution": dict(expression_counts)
        }
    
    async def get_attention_trend(self) -> Dict:
        """Get attention trend over time"""
        if len(self.expression_history) < 5:
            return {"trend": "insufficient_data", "change": 0, "current_attention": 0.5}
        
        recent_scores = [e['attention_score'] for e in self.expression_history[-10:]]
        
        if len(recent_scores) > 5:
            recent_avg = np.mean(recent_scores[-5:])
            previous_avg = np.mean(recent_scores[-10:-5])
            change = recent_avg - previous_avg
        else:
            change = 0
        
        if change > 0.1:
            trend = "improving"
        elif change < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": round(change, 3),
            "current_attention": round(recent_scores[-1], 3) if recent_scores else 0.5,
            "history": [round(s, 3) for s in recent_scores[-20:]]
        }