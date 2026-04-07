# 🦊 Rapid Foxin - AI Student Productivity System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

## 🚀 Vision

**Rapid Foxin** is a production-grade AI system that transforms student productivity through real-time multimodal perception and reinforcement learning. Think Jarvis meets an AI coach - continuously observing, predicting, and optimizing your focus state.

## 🎯 Problem Statement

Students struggle with maintaining focus in a distraction-rich environment. Traditional productivity tools are reactive. Rapid Foxin is **predictive** and **adaptive** - it learns your patterns and intervenes before distraction occurs.

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐

│ Rapid Foxin Core │

├──────────────┬──────────────┬──────────────┬────────────────┤

│ Perception │ RL Engine  │ API Layer │ Dashboard │

│ ┌────────┐ │ ┌────────┐ │ ┌────────┐ │ ┌──────────┐ │

│ │Screen │  │ │ DQN    │ │ │FastAPI │ │ │Jarvis UI│ │

│ │Intel │ │ │ Agent    │ │ │WebSocket│ │ │Real-time │ │

│ └────────┘ │ └────────┘ │ └────────┘ │ │ Charts │ │

│ ┌────────┐ │ ┌────────┐ │ ┌────────┐ │ └──────────┘ │

│ │Camera │ │ │Replay │ │ │Models │ │ │

│ │Intel │ │ │Buffer │ │ │ │ │ │

│ └────────┘ │ └────────┘ │ └────────┘ │ │

│ ┌────────┐ │ ┌────────┐ │ │ │

│ │Fusion │ │ │Target │ │ │ │

│ │Engine │ │ │Network │ │ │ │

│ └────────┘ │ └────────┘ │ │ │

└──────────────┴──────────────┴───────────────┴────────────────┘


## 🧠 Reinforcement Learning System

### States
- **Focused** - Productive work state
- **Distracted** - Off-task behavior  
- **Tired** - Fatigue accumulation
- **Deep Focus** - Optimal performance (bonus state)

### Actions
- Study - Continue working
- Take Break - Rest period
- Use Phone - Distraction action
- Switch Task - Task transition

## 📊 Features

- **Real-time Focus Scoring** (0-100)
- **Predictive Attention Tracking**
- **Adaptive RL Training**
- **WebSocket Live Updates**
- **Jarvis-style Dashboard**
- **Docker Containerization**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Uvicorn |
| RL Engine | PyTorch, NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js |
| Container | Docker |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rapid-foxin.git
cd rapid-foxin

# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Build the image
docker build -t rapid-foxin .

# Run the container
docker run -p 8000:8000 rapid-foxin
