STATE_FILE = "state.json"

# 졸음 감지 설정
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # 눈 감김 판단 비율 (조정 필요)
FRAME_WINDOW = 15                  # 최근 15프레임 분석
CLOSED_EYE_THRESHOLD = 10          # 15프레임 중 10번 이상 감기면 졸음

# 감정 레이블
EMOTIONS = ["anger", "closed", "happy", "panic", "sadness"]
EMOTION_SMOOTHING_WINDOW = 10 # 최근 10프레임의 감정을 분석

# Gemini API 키 (여기에 본인의 키를 입력하세요)
GOOGLE_API_KEY = "REMOVED"