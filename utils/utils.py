# utils.py
import json
import time
import os
from config import STATE_FILE


def save_state(emotion, is_accident):
    """(시뮬레이터용) 현재 상태를 파일에 저장"""
    data = {
        "emotion": emotion,
        "is_accident": is_accident,
        "timestamp": time.time()
    }
    try:
        with open(STATE_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"상태 저장 오류: {e}")


def load_state():
    """(Streamlit용) 파일에서 상태 읽기"""
    if not os.path.exists(STATE_FILE):
        return "happy", False

    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
            return (
                data.get("emotion", "happy"),
                data.get("is_accident", False)
            )
    except:
        return "happy", False
