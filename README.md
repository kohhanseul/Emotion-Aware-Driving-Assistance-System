# Emotion-Aware Driving Assistant (CHADURI)

## Project Overview
CHADURI is an emotion-aware driving assistance system that recognizes a driver’s facial expressions in real time and provides appropriate safety actions based on the detected emotional state.

Unlike conventional ADAS systems that rely mainly on vehicle behavior, this project focuses on **driver emotional awareness** as a primary safety signal, enabling earlier risk detection and context-aware responses.

---

## Problem Statement
Traffic accidents caused by driver factors such as drowsiness, anger, panic, and impulsive behavior remain a major contributor to road fatalities.

This project aims to:
- Detect driver emotions directly from facial expressions
- Distinguish normal behavior (e.g., blinking) from dangerous states (e.g., microsleep)
- Trigger emotion-specific assistance actions to prevent accidents

---

## Dataset & Class Design
- Original facial expression dataset was restructured and cleaned
- Ambiguous and low-quality samples were **manually reviewed and removed**
- Final emotion classes:
  - **Anger** (aggressive / risky driving)
  - **Closed** (drowsiness / microsleep)
  - **Panic** (sudden emergency response)
  - **Sadness** (low attention / risk)
  - **Happy** (safe driving state)

To improve drowsiness detection accuracy:
- Normal blink duration (0.1–0.4s) was distinguished from microsleep (≥2s)
- Frame-based temporal consistency was considered

---

## Model Development & Experiments

### 1. Baseline Models
- CNN-based pretrained models:
  - VGG16, ResNet50, MobileNet, EfficientNet, DenseNet
- Initial evaluation without fine-tuning

### 2. Fine-Tuning Strategy
- Layer-wise fine-tuning experiments:
  - Top-20 layers
  - Top-30%
  - Full fine-tuning
- Best **baseline** performance achieved with **EfficientNet fine-tuned at top-250 layers**

### 3. Feature Extraction + ML
- Deep features extracted from:
  - Dense layer
  - Global Average Pooling (GAP)
- Linear SVC achieved best generalization without StandardScaler

---

## Custom Model: EffEmoteNet
To improve both accuracy and real-time performance, a custom model was designed.

### Key Design Choices
- Base architecture inspired by EfficientNet and ResEmoteNet
- **MBConv blocks** used instead of standard residual blocks for parameter efficiency
- **Sobel edge channel** added to emphasize eye-closure patterns
- **CBAM (Channel & Spatial Attention)** applied to focus on critical facial regions

### Result
EffEmoteNet achieved the best balance between:
- Accuracy
- Model size
- Inference speed

---

## Real-Time Performance Evaluation
- Inference speed comparison:
  - ViT / EffEmoteNet: **4–8 ms per frame**
  - EfficientNetV2-S: ~37 ms per frame
- At 30 FPS:
  - EffEmoteNet reliably distinguishes normal blinking (3–12 frames)
    from microsleep (~60 frames)
  - Suitable for real-time driving assistance

---

## System Architecture

### Demo Environment
- **Webcam**: driver face input
- **Emotion Classification Model**: EffEmoteNet + Sobel + CBAM
- **LLM (Gemini 2.5 Flash)**:
  - Emotion explanation
  - Driver interaction messages
- **Streamlit**: real-time dashboard
- **MetaDrive Simulator**: emotion-driven vehicle behavior simulation

---

## Assistance Scenarios
*(The following scenarios are partially implemented for demo purposes.)*

- **Closed (Drowsy)**:
  - Hazard lights ON
  - Shoulder stop
  - Interactive word-chain game for alertness
- **Panic**:
  - Emergency message
  - Location sharing if no driver response
- **Anger / Sadness**:
  - Caution warnings to surrounding vehicles
- **Happy**:
  - Safe driving indicator

---

## Tech Stack
- **Language**: Python
- **DL Frameworks**: TensorFlow, PyTorch
- **CV**: OpenCV, MTCNN
- **Models**: EfficientNet, ViT, Custom CNN (EffEmoteNet)
- **Attention**: CBAM
- **Simulation**: MetaDrive
- **UI**: Streamlit
- **LLM API**: Gemini

---

## My Role
- Designed overall system architecture
- Conducted extensive model comparison and fine-tuning experiments
- **Led the design and implementation of the EffEmoteNet custom model**
- Integrated Sobel filtering and CBAM attention
- Implemented real-time inference and demo system
- Connected emotion recognition to driving simulation and LLM-based interaction

---

## Notes
- This repository represents a **demo implementation** of the proposed system.
- Core components such as real-time emotion recognition, model inference,
  and UI integration are fully implemented.
- Assistance scenarios are designed at the system level and partially implemented
  to validate feasibility and real-time performance.
