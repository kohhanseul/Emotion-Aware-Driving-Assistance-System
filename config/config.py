"""프로젝트 전역 설정.

감지 임계값, 감정 레이블, API 키(환경변수) 등을 관리한다.
API 키는 프로젝트 루트의 .env 파일에서 읽어온다 (.env.example 참고).
"""
import os
from dotenv import load_dotenv

load_dotenv()

STATE_FILE = "state.json"

# 졸음 감지 설정
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # 눈 감김 판단 비율
FRAME_WINDOW = 15                  # 최근 15프레임 분석
CLOSED_EYE_THRESHOLD = 10          # 15프레임 중 10번 이상 감기면 졸음

# 감정 레이블
EMOTIONS = ["anger", "closed", "happy", "panic", "sadness"]
EMOTION_SMOOTHING_WINDOW = 10      # 최근 10프레임의 감정을 분석

# API 키 (환경변수에서 로드 — 코드에 직접 쓰지 말 것)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
KOREAN_DICT_API_KEY = os.getenv("KOREAN_DICT_API_KEY", "")
