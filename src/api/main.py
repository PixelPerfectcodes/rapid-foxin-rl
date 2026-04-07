from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import base64
import asyncio

from src.env.productivity_env import ProductivityEnv
from src.agents.dqn_agent import DQNAgent
from src.perception.fusion_engine import FusionEngine
from src.perception.camera_intel import CameraIntelligence
from src.api.models import ActionRequest, ResetRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
env = None
agent = None
fusion_engine = None
camera_intel = None
action_names = []
active_websockets: List[WebSocket] = []
training_episodes = 0
total_rewards = []
episode_rewards = []
focus_history = []
reward_history = []
streak_history = []
camera_active = False
emotion_history = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global env, agent, fusion_engine, camera_intel, action_names
    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING RAPID FOXIN AI SYSTEM")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("📦 Loading Environment...")
        env = ProductivityEnv()
        
        logger.info("🧠 Initializing Fusion Engine...")
        fusion_engine = FusionEngine()
        
        logger.info("📷 Setting up Camera Intelligence with Facial Expression Detection...")
        camera_intel = CameraIntelligence()
        
        logger.info("🤖 Creating DQN Agent...")
        action_names = env.action_names
        state_dim = len(env.get_state()['vector'])
        action_dim = len(action_names)
        agent = DQNAgent(state_dim, action_dim)
        
        logger.info("✅ System Initialization Complete!")
        logger.info(f"   - States: {len(env.state_names)}")
        logger.info(f"   - Actions: {action_dim}")
        logger.info(f"   - State Dimension: {state_dim}")
        logger.info(f"   - WebSocket Ready: Yes")
        logger.info(f"   - Camera Ready: Yes")
        logger.info(f"   - OCR Ready: Yes")
        logger.info(f"   - Expression Detection: Yes")
        
    except Exception as e:
        logger.error(f"❌ Initialization Error: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("=" * 60)
    logger.info("🛑 SHUTTING DOWN RAPID FOXIN AI SYSTEM")
    logger.info("=" * 60)
    
    if camera_intel:
        await camera_intel.stop_camera()
        logger.info("✓ Camera stopped")
    
    if active_websockets:
        for ws in active_websockets:
            await ws.close()
        logger.info(f"✓ Closed {len(active_websockets)} WebSocket connections")
    
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Rapid Foxin AI",
    description="Intelligent Student Productivity Platform with Facial Expression Detection",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_focus_score(state: dict) -> float:
    """Calculate focus score from environment state"""
    base_score = {
        'deep_focus': 95,
        'focused': 75,
        'neutral': 50,
        'distracted': 30,
        'tired': 15
    }.get(state['current_state'], 50)
    
    fatigue_penalty = state['fatigue'] * 20
    streak_bonus = min(15, state['focus_streak'] * 0.5)
    
    score = base_score - fatigue_penalty + streak_bonus
    return max(0, min(100, score))

async def broadcast_state_update(state: dict, reward: float, done: bool):
    """Broadcast state update to all connected WebSocket clients"""
    if not active_websockets:
        return
    
    message = {
        "type": "env_update",
        "state": {
            "current_state": state['current_state'],
            "fatigue": state['fatigue'],
            "focus_streak": state['focus_streak'],
            "focus_score": calculate_focus_score(state)
        },
        "reward": reward,
        "done": done,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

async def broadcast_camera_update(camera_data: dict):
    """Broadcast camera analysis to all connected WebSocket clients"""
    if not active_websockets:
        return
    
    message = {
        "type": "camera_update",
        "data:": camera_data,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def get_root():
    """Serve the main dashboard"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return JSONResponse({
            "status": "error",
            "message": "Frontend not found. Please ensure frontend/index.html exists."
        }, status_code=404)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "system": "Rapid Foxin AI",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_websockets),
        "camera_active": camera_active,
        "ocr_available": True,
        "opencv_available": True,
        "expression_detection": True,
        "environment": {
            "current_state": env.state.current_state if env else "unknown",
            "steps": env.state.step_count if env else 0
        },
        "agent": {
            "epsilon": agent.epsilon if agent else 1.0,
            "memory_size": len(agent.memory) if agent else 0
        }
    })

# ============================================================================
# CAMERA ENDPOINTS
# ============================================================================

@app.post("/camera/start")
async def start_camera():
    """Start the webcam for face detection and expression recognition"""
    global camera_active
    try:
        success = await camera_intel.start_camera()
        if success:
            camera_active = True
            logger.info("✅ Camera started successfully with expression detection")
            await broadcast_camera_update({
                "status": "started",
                "message": "Camera active with facial expression detection"
            })
            return JSONResponse({
                "status": "success",
                "message": "Camera started. Face detection, expression recognition, and OCR active.",
                "camera_active": True,
                "features": ["face_detection", "expression_recognition", "ocr", "attention_tracking"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Failed to start camera. Please check camera permissions."
            }, status_code=500)
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/camera/stop")
async def stop_camera():
    """Stop the webcam"""
    global camera_active
    try:
        await camera_intel.stop_camera()
        camera_active = False
        logger.info("Camera stopped")
        await broadcast_camera_update({
            "status": "stopped",
            "message": "Camera deactivated"
        })
        return JSONResponse({
            "status": "success",
            "message": "Camera stopped.",
            "camera_active": False
        })
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/camera/analyze")
async def analyze_camera_frame(file: UploadFile = File(None)):
    """Analyze camera frame for face detection, expression, and text extraction"""
    try:
        if file:
            # Read uploaded image file
            contents = await file.read()
            result = await camera_intel.analyze_frame(contents)
        elif camera_active:
            # Use live camera feed
            result = await camera_intel.analyze_frame()
        else:
            # Camera not active
            result = camera_intel._default_response()
        
        return JSONResponse({
            "status": "success",
            "data": result
        })
    except Exception as e:
        logger.error(f"Error analyzing camera frame: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/camera/analyze-base64")
async def analyze_base64_frame(data: dict):
    """Analyze camera frame from base64 string"""
    try:
        base64_string = data.get('frame', '')
        if not base64_string:
            return JSONResponse({
                "status": "error",
                "message": "No frame data provided"
            }, status_code=400)
        
        result = await camera_intel.analyze_frame_from_base64(base64_string)
        
        # Store in global emotion history
        global emotion_history
        if result.get('face_detected'):
            emotion_history.append({
                "timestamp": datetime.now().isoformat(),
                "expression": result.get('overall_expression', 'Unknown'),
                "attention_score": result.get('attention_score', 0),
                "engagement_score": result.get('engagement_score', 0),
                "focus_score": result.get('focus_score', 0)
            })
            if len(emotion_history) > 100:
                emotion_history = emotion_history[-100:]
        
        return JSONResponse({
            "status": "success",
            "data": result
        })
    except Exception as e:
        logger.error(f"Error analyzing base64 frame: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/camera/expression-summary")
async def get_expression_summary():
    """Get summary of detected facial expressions"""
    try:
        if not camera_active:
            # Return history even if camera not active
            if emotion_history:
                summary = await camera_intel.get_expression_summary() if camera_intel else {"has_data": False}
            else:
                return JSONResponse({
                    "status": "success",
                    "data": {"has_data": False, "message": "No expression data available"}
                })
        else:
            summary = await camera_intel.get_expression_summary()
        
        return JSONResponse({
            "status": "success",
            "data": summary
        })
    except Exception as e:
        logger.error(f"Error getting expression summary: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/camera/emotion-timeline")
async def get_emotion_timeline(limit: int = 50):
    """Get emotion timeline for recent frames"""
    try:
        if camera_active and camera_intel:
            timeline_data = []
            for entry in camera_intel.expression_history[-limit:]:
                timeline_data.append({
                    "timestamp": entry['timestamp'].isoformat() if hasattr(entry['timestamp'], 'isoformat') else str(entry['timestamp']),
                    "expression": entry.get('expression', 'Unknown'),
                    "attention_score": entry.get('attention_score', 0),
                    "engagement_score": entry.get('engagement_score', 0)
                })
        else:
            # Return from global history
            timeline_data = emotion_history[-limit:]
        
        return JSONResponse({
            "status": "success",
            "data": timeline_data,
            "count": len(timeline_data)
        })
    except Exception as e:
        logger.error(f"Error getting emotion timeline: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/camera/text-summary")
async def get_text_summary():
    """Get summary of extracted text from recent frames"""
    try:
        if not camera_active:
            return JSONResponse({
                "status": "error",
                "message": "Camera not active"
            }, status_code=400)
        
        summary = await camera_intel.get_text_summary()
        return JSONResponse({
            "status": "success",
            "data": summary
        })
    except Exception as e:
        logger.error(f"Error getting text summary: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/camera/attention-trend")
async def get_attention_trend():
    """Get attention trend over time"""
    try:
        if not camera_active:
            return JSONResponse({
                "status": "error",
                "message": "Camera not active"
            }, status_code=400)
        
        trend = await camera_intel.get_attention_trend()
        return JSONResponse({
            "status": "success",
            "data": trend
        })
    except Exception as e:
        logger.error(f"Error getting attention trend: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/camera/status")
async def get_camera_status():
    """Get camera status and current metrics"""
    global camera_active
    try:
        if camera_active:
            attention_trend = await camera_intel.get_attention_trend()
            text_summary = await camera_intel.get_text_summary()
            expression_summary = await camera_intel.get_expression_summary()
            
            return JSONResponse({
                "status": "success",
                "camera_active": True,
                "attention_trend": attention_trend,
                "text_summary": text_summary,
                "expression_summary": expression_summary,
                "features": ["face_detection", "expression_recognition", "ocr", "attention_tracking"]
            })
        else:
            return JSONResponse({
                "status": "success",
                "camera_active": False,
                "message": "Camera not active. Start camera to enable face detection, expression recognition, and OCR."
            })
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ============================================================================
# ENVIRONMENT ENDPOINTS
# ============================================================================

@app.get("/state")
async def get_state():
    """Get current environment state with camera data"""
    try:
        state = env.get_state()
        
        # Get current camera analysis if active
        camera_data = None
        if camera_active:
            camera_data = await camera_intel.analyze_frame()
        
        return JSONResponse({
            "status": "success",
            "state": {
                "current_state": state['current_state'],
                "fatigue_level": state['fatigue'],
                "focus_streak": state['focus_streak'],
                "attention_drift": state['attention_drift'],
                "step_count": env.state.step_count,
                "focus_score": calculate_focus_score(state)
            },
            "camera_data": camera_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/step")
async def step_action(request: ActionRequest):
    """Execute an action in the environment"""
    global episode_rewards, training_episodes, total_rewards, focus_history, reward_history, streak_history
    
    try:
        action_name = action_names[request.action]
        
        # Get AI focus score from camera if active
        ai_score = request.ai_focus_score if request.ai_focus_score else 75.0
        camera_expression = None
        
        if camera_active:
            camera_result = await camera_intel.analyze_frame()
            ai_score = camera_result.get('focus_score', 75.0)
            camera_expression = camera_result.get('overall_expression', 'Unknown')
        
        # Execute step in environment
        state, reward, done, info = env.step(action_name, ai_score)
        
        # Train agent
        state_vector = state['vector']
        next_state = env.get_state()
        next_state_vector = next_state['vector']
        
        agent.remember(state_vector, request.action, reward, next_state_vector, done)
        loss = agent.train()
        
        # Track rewards
        episode_rewards.append(reward)
        
        if done:
            training_episodes += 1
            total_rewards.append(sum(episode_rewards))
            episode_rewards = []
            logger.info(f"Episode {training_episodes} completed - Total Reward: {total_rewards[-1]:.2f}")
        
        # Update histories
        focus_score = calculate_focus_score(state)
        focus_history.append(focus_score)
        reward_history.append(reward)
        streak_history.append(state['focus_streak'])
        
        # Keep only last 100 points
        if len(focus_history) > 100:
            focus_history.pop(0)
        if len(reward_history) > 100:
            reward_history.pop(0)
        if len(streak_history) > 100:
            streak_history.pop(0)
        
        # Broadcast to WebSocket clients
        await broadcast_state_update(state, reward, done)
        
        # Log action
        logger.info(f"Action: {action_name} | Reward: {reward:.2f} | State: {state['current_state']} | Focus: {focus_score:.1f} | Expression: {camera_expression}")
        
        return JSONResponse({
            "status": "success",
            "state": {
                "current_state": state['current_state'],
                "fatigue": state['fatigue'],
                "focus_streak": state['focus_streak'],
                "attention_drift": state['attention_drift'],
                "focus_score": focus_score
            },
            "reward": reward,
            "done": done,
            "info": info,
            "loss": loss,
            "camera_expression": camera_expression
        })
        
    except Exception as e:
        logger.error(f"Error in step_action: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/reset")
async def reset_environment(request: ResetRequest = None):
    """Reset the environment"""
    try:
        state = env.reset()
        global episode_rewards
        episode_rewards = []
        
        logger.info("Environment reset")
        
        return JSONResponse({
            "status": "success",
            "state": {
                "current_state": state['current_state'],
                "fatigue": state['fatigue'],
                "focus_streak": state['focus_streak'],
                "attention_drift": state['attention_drift'],
                "focus_score": 75.0
            },
            "message": "Environment reset successfully"
        })
    except Exception as e:
        logger.error(f"Error resetting environment: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/metrics")
async def get_metrics():
    """Get training metrics"""
    try:
        avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0
        avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
        
        return JSONResponse({
            "status": "success",
            "episodes": training_episodes,
            "avg_reward": float(avg_reward),
            "avg_loss": float(avg_loss),
            "epsilon": agent.epsilon,
            "memory_size": len(agent.memory),
            "focus_streak": env.state.focus_streak,
            "fatigue_level": env.state.fatigue_level,
            "focus_history": focus_history[-50:],
            "reward_history": reward_history[-50:],
            "streak_history": streak_history[-50:],
            "camera_active": camera_active,
            "total_steps": env.state.step_count
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/focus-score")
async def get_focus_score():
    """Get current focus score and trend with camera integration"""
    try:
        state = env.get_state()
        focus_score = calculate_focus_score(state)
        
        # Get camera attention if active
        camera_attention = None
        extracted_text = None
        text_activity = None
        current_expression = None
        
        if camera_active:
            camera_result = await camera_intel.analyze_frame()
            camera_attention = camera_result.get('attention_score', 0.5)
            extracted_text = camera_result.get('extracted_text', '')
            text_activity = camera_result.get('text_classification', {}).get('label', 'unknown')
            current_expression = camera_result.get('overall_expression', 'Unknown')
        
        # Calculate trend
        trend = "stable"
        if len(focus_history) > 5:
            recent = focus_history[-5:]
            if recent[-1] > recent[0] + 5:
                trend = "improving"
            elif recent[-1] < recent[0] - 5:
                trend = "declining"
        
        return JSONResponse({
            "status": "success",
            "focus_score": focus_score,
            "trend": trend,
            "state": state['current_state'],
            "camera_attention": camera_attention,
            "camera_active": camera_active,
            "text_activity": text_activity,
            "current_expression": current_expression,
            "extracted_text_preview": extracted_text[:200] if extracted_text else None,
            "focus_streak": state['focus_streak'],
            "fatigue_level": state['fatigue']
        })
    except Exception as e:
        logger.error(f"Error getting focus score: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/train-dqn")
async def train_dqn(episodes: int = 50):
    """Train the DQN agent"""
    global training_episodes, total_rewards
    
    try:
        logger.info(f"Starting DQN training for {episodes} episodes...")
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            state_vector = state['vector']
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 1000:
                action = agent.act(state_vector)
                action_name = action_names[action]
                next_state, reward, done, _ = env.step(action_name, 75.0)
                
                next_state_vector = next_state['vector']
                agent.remember(state_vector, action, reward, next_state_vector, done)
                agent.train()
                
                state_vector = next_state_vector
                episode_reward += reward
                step_count += 1
            
            rewards.append(episode_reward)
            total_rewards.append(episode_reward)
            training_episodes += 1
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                logger.info(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        
        avg_reward = np.mean(rewards)
        avg_score = min(100, max(0, (avg_reward + 50) / 2))  # Normalize to 0-100
        
        logger.info(f"Training completed! Avg Reward: {avg_reward:.2f} - Score: {avg_score:.1f}")
        
        return JSONResponse({
            "status": "success",
            "average_reward": float(avg_reward),
            "average_score": float(avg_score),
            "episodes": episodes,
            "final_epsilon": agent.epsilon,
            "message": f"Training completed successfully over {episodes} episodes"
        })
    except Exception as e:
        logger.error(f"Error training DQN: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates with camera and expression data"""
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info(f"🔌 WebSocket connected. Total connections: {len(active_websockets)}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get('type', 'unknown')
            
            if message_type == 'perception':
                # Get current state
                state = env.get_state()
                focus_score = calculate_focus_score(state)
                
                # Get camera analysis if active
                camera_data = None
                if camera_active:
                    camera_result = await camera_intel.analyze_frame()
                    camera_data = {
                        "attention_score": camera_result.get('attention_score', 0.5),
                        "expression": camera_result.get('overall_expression', 'unknown'),
                        "face_detected": camera_result.get('face_detected', False),
                        "face_count": camera_result.get('face_count', 0),
                        "extracted_text": camera_result.get('extracted_text', '')[:200],
                        "text_classification": camera_result.get('text_classification', {}),
                        "focus_confidence": camera_result.get('focus_score', 50),
                        "engagement_score": camera_result.get('engagement_score', 0),
                        "emotions_detected": camera_result.get('expressions_detected', [])
                    }
                
                # Simulate screen data
                screen_data = {"productivity_score": focus_score / 100}
                
                # Fuse signals
                fused = await fusion_engine.fuse_signals(
                    screen_data,
                    camera_result if camera_active else {"focus_confidence": 50},
                    state
                )
                
                # Send AI state update
                await websocket.send_json({
                    "type": "ai_state",
                    "data": fused,
                    "camera_data": camera_data,
                    "prediction": fusion_engine.get_prediction(),
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == 'camera_frame':
                # Process camera frame from frontend
                frame_data = message.get('frame')
                if frame_data:
                    result = await camera_intel.analyze_frame_from_base64(frame_data)
                    
                    # Store in emotion history
                    if result.get('face_detected'):
                        emotion_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "expression": result.get('overall_expression', 'Unknown'),
                            "attention_score": result.get('attention_score', 0),
                            "engagement_score": result.get('engagement_score', 0),
                            "focus_score": result.get('focus_score', 0)
                        })
                        if len(emotion_history) > 100:
                            emotion_history.pop(0)
                    
                    await websocket.send_json({
                        "type": "camera_analysis",
                        "data": result,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == 'action':
                # Execute action from client
                action = message.get('action')
                if action in action_names:
                    # Get AI score from camera if active
                    ai_score = 75.0
                    camera_expression = None
                    
                    if camera_active:
                        camera_result = await camera_intel.analyze_frame()
                        ai_score = camera_result.get('focus_score', 75.0)
                        camera_expression = camera_result.get('overall_expression', 'Unknown')
                    
                    state, reward, done, _ = env.step(action, ai_score)
                    
                    await websocket.send_json({
                        "type": "env_update",
                        "state": {
                            "current_state": state['current_state'],
                            "fatigue": state['fatigue'],
                            "focus_streak": state['focus_streak'],
                            "focus_score": calculate_focus_score(state)
                        },
                        "reward": reward,
                        "done": done,
                        "camera_expression": camera_expression,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == 'get_expression_history':
                # Send expression history
                await websocket.send_json({
                    "type": "expression_history",
                    "data": emotion_history[-50:],
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == 'ping':
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total connections: {len(active_websockets)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# ============================================================================
# STATIC FILES
# ============================================================================

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    logger.info("✅ Static files mounted from /frontend")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# ============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# ============================================================================

@app.get("/system-info")
async def get_system_info():
    """Get detailed system information"""
    return JSONResponse({
        "status": "success",
        "system": {
            "name": "Rapid Foxin AI",
            "version": "3.0.0",
            "framework": "FastAPI",
            "rl_algorithm": "DQN (Deep Q-Network)",
            "perception": ["OpenCV", "Tesseract OCR", "MediaPipe"],
            "features": ["Face Detection", "Expression Recognition", "OCR", "Attention Tracking", "RL Training"]
        },
        "environment": {
            "states": env.state_names if env else [],
            "actions": action_names,
            "max_steps": env.params.get('max_steps', 1000) if env else 0
        },
        "agent": {
            "epsilon": agent.epsilon if agent else 1.0,
            "memory_capacity": len(agent.memory) if agent else 0,
            "training_episodes": training_episodes
        },
        "camera": {
            "active": camera_active,
            "features": ["expression_detection", "ocr", "face_tracking"]
        },
        "websocket": {
            "active_connections": len(active_websockets)
        }
    })

@app.get("/clear-history")
async def clear_history():
    """Clear all history data"""
    global focus_history, reward_history, streak_history, emotion_history, episode_rewards, total_rewards
    focus_history = []
    reward_history = []
    streak_history = []
    emotion_history = []
    episode_rewards = []
    total_rewards = []
    
    logger.info("History cleared")
    
    return JSONResponse({
        "status": "success",
        "message": "All history data cleared"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )