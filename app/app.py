import time
import re
import random
import requests
import os
import streamlit as st

# LangChain 및 Gemini 관련 임포트
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 프로젝트 설정 및 유틸리티 임포트
import config
from utils import load_state

# ==========================================
# 1. 초기 설정 (API 키 및 세션)
# ==========================================
# Gemini API 키 설정 (config.py에서 가져오거나 직접 설정)
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY

# 국어사전 API 키 config에서 가져오기 (상단 import 부근):
from config.config import GOOGLE_API_KEY, KOREAN_DICT_API_KEY

# Streamlit 페이지 설정
st.set_page_config(page_title="차두리 AI", layout="centered")

# 세션 상태 초기화
if "word_game_active" not in st.session_state:
    st.session_state.update({
        "word_game_active": False,
        "word_game_used_words": [],
        "word_game_last_word": None,
        "word_game_message": "",
        "word_game_turn": 0,
        "prev_emotion": None,
        "auto_refresh_enabled": False,  # 자동 새로고침 토글 상태
        "current_emotion": "neutral",
        "is_accident_state": False,
    })

# ==========================================
# 2. 로직 함수들 (노트북 내용 복원)
# ==========================================

# --- LLM 설정 ---
try:
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05",  # 최신 모델명 권장 (혹은 노트북의 모델 사용)
        temperature=0.4,
    )
except Exception:
    # 모델명 오류 시 fallback
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
    )

# --- 차두리 Next Action 판단 ---
emotion_template = """
너는 운전 보조 AI '차두리'야.
운전자는 먼저 말하지 않고, 너는 운전자의 감정에 따라 정해진 next_action만 해

[입력]
- 감정: {emotion}      
- 상황: {situation}    

[감정별 규칙]
1) anger (화남) -> "GUIDE_MEDITATION"
2) closed (졸림) -> "ASK_WORD_GAME" (국어사전 단어만 사용)
3) happy (기분 좋음) -> "ASK_CALM_MUSIC"
4) panic (패닉/불안) -> "VERIFY_EMERGENCY"
5) sadness (슬픔) -> "ASK_UPBEAT_MUSIC"

[출력 형식]
JSON 형식으로만 대답해:
{{ "next_action": "ACTION_NAME" }}
"""

parser = JsonOutputParser()
emotion_prompt = PromptTemplate(
    template=emotion_template,
    input_variables=["emotion", "situation"],
)
emotion_chain = emotion_prompt | model | parser


def get_next_action(emotion: str, situation: str) -> str:
    try:
        result = emotion_chain.invoke({"emotion": emotion, "situation": situation})
        if isinstance(result, dict):
            return result.get("next_action", "NONE")
        return str(result)
    except Exception as e:
        print(f"LLM Error: {e}")
        return "NONE"


# --- 끝말잇기 관련 로직 ---
WORD_CACHE: dict[str, bool] = {}
HANGUL_REGEX = re.compile(r"^[가-힣]+$")


def is_real_korean_word(word: str) -> bool:
    if word in WORD_CACHE:
        return WORD_CACHE[word]
    params = {
        "key": KOREAN_DICT_API_KEY,
        "q": word,
        "req_type": "json",
        "advanced": "y",
        "type1": "word",
        "method": "exact",
        "num": 10,
    }
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        channel = data.get("channel", {})
        total = int(channel.get("total", 0))
        exists = total > 0
    except Exception as e:
        print("사전 API 호출 중 오류:", e)
        exists = False  # 오류나면 없는 단어로 취급
    WORD_CACHE[word] = exists
    return exists


def get_word_candidates(char: str, used_words):
    used_str = ", ".join(used_words) if used_words else "없음"
    prompt_template = PromptTemplate.from_template(
        """
        {char} 로 시작하는 한국어 명사 10개를 나열해.
        - 국어사전에 있는 단어여야 해.
        - 이미 사용된 단어 제외: {used_str}
        - 출력: 단어만 한 줄에 하나씩
        """
    )
    res = model.invoke(prompt_template.format(char=char, used_str=used_str))
    raw = res.content.strip()
    return [w.strip() for w in raw.splitlines() if w.strip()]


def pick_valid_word(candidates, char, used_words):
    for word in candidates:
        if not HANGUL_REGEX.match(word) or len(word) < 2: continue
        if not word.startswith(char): continue
        if word in used_words: continue
        if is_real_korean_word(word): return word
    return None


# --- 음악 리스트 ---
calm_music_list = ["10CM - 너에게 닿기를", "DAY6 - 예뻤어", "백예린 - square", "아이유 - 밤편지", "성시경 - 거리에서"]
upbeat_music_list = ["BTS - Dynamite", "BLACKPINK - 붐바야", "ZICO - 아무노래", "NewJeans - Super Shy", "싸이 - 챔피언"]


# ==========================================
# 3. UI 렌더링 함수들
# ==========================================

def render_external_display(external_signal, color):
    st.markdown(
        f"""
        <div style="
            background-color:#111;
            border-radius:24px;
            padding:20px 10px;
            text-align:center;
            border: 4px solid {color};
            margin-bottom: 20px;
        ">
            <div style="font-size:40px; color:{color}; font-weight:bold;">{external_signal}</div>
            <div style="font-size:16px; color:#ccc; margin-top:10px;">차량 후면 디스플레이</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meditation_ui():
    st.subheader("🧘 차두리 심호흡 가이드")
    if st.button("심호흡 시작"):
        placeholder = st.empty()
        placeholder.write("🧘‍♀️ 분노의 조짐이 보여요! 잠시 심호흡 시간을 가져볼게요.")
        time.sleep(2)
        for i in range(3):
            placeholder.write(f"💨 천천히 숨을 들이쉬고... ({i + 1}/3)")
            time.sleep(2)
            placeholder.write(f"💨 길게 내쉬세요. ({i + 1}/3)")
            time.sleep(2)
        placeholder.success("마음이 좀 진정되셨나요? 안전운전 하세요! 🚗")


def render_word_game_ui():
    st.subheader("🎮 졸음 깨기 끝말잇기")

    if st.session_state.word_game_message:
        st.info(st.session_state.word_game_message)

    with st.form("word_game_form"):
        user_word = st.text_input("단어 입력 (종료하려면 '그만'):")
        submit = st.form_submit_button("입력")

    if submit:
        if user_word == "그만":
            st.session_state.word_game_active = False
            st.rerun()

        # 유효성 검사 로직
        used = st.session_state.word_game_used_words
        last = st.session_state.word_game_last_word

        valid = True
        msg = ""

        if not HANGUL_REGEX.match(user_word) or len(user_word) < 2:
            valid = False
            msg = "❌ 한글 2글자 이상이어야 해요!"
        elif last and not user_word.startswith(last[-1]):
            valid = False
            msg = f"❌ '{last[-1]}'로 시작해야 해요!"
        elif user_word in used:
            valid = False
            msg = "❌ 이미 쓴 단어예요!"
        elif not is_real_korean_word(user_word):
            valid = False
            msg = "❌ 사전에 없는 단어예요!"

        if not valid:
            st.session_state.word_game_message = msg + " (당신 패배! 🤪)"
            st.session_state.word_game_active = False  # 게임 리셋
            st.rerun()

        # 사용자 성공
        used.append(user_word)
        st.session_state.word_game_used_words = used

        # AI 차례
        last_char = user_word[-1]
        with st.spinner("AI가 생각 중..."):
            candidates = get_word_candidates(last_char, used)
            ai_word = pick_valid_word(candidates, last_char, used)

        if ai_word:
            used.append(ai_word)
            st.session_state.word_game_last_word = ai_word
            st.session_state.word_game_message = f"나: {user_word} -> 차두리: {ai_word}"
        else:
            st.session_state.word_game_message = f"당신 승리! 🎉 '{last_char}'로 시작하는 단어를 못 찾겠어요."
            st.session_state.word_game_active = False

        st.rerun()


def render_music_ui(music_list, title):
    st.subheader(f"🎵 {title}")
    if st.button("음악 추천 받기"):
        song = random.choice(music_list)
        st.success(f"추천 곡: {song}")
        st.caption("음악이 재생됩니다... (상상해보세요 🎧)")


def render_verify_emergency_ui():
    st.subheader("🚨 응급 상황 확인")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚨 응급 신고 (119)", type="primary"):
            st.toast("119에 위치 전송 완료!")
            st.error("구조대가 출동했습니다.")
    with col_b:
        if st.button("괜찮아요"):
            st.info("다행입니다. 안전한 곳에 정차해 주세요.")


# ==========================================
# 4. 메인 앱 실행
# ==========================================

st.title("🚘 차두리 (AI Driver Assistant)")
st.caption("Vision AI & Simulator Connected System")

# 사이드바 설정
st.sidebar.header("설정")
mode = st.sidebar.radio("모드 선택", ["Manual (테스트)", "Auto (센서 연동)"])

# 1. 상태 결정 로직 초기화 (모든 경우에 대비한 안전한 기본값)
current_emotion = "neutral"
is_accident = False

# 2. 모드별 위젯 렌더링 및 값 결정
if mode == "Manual (테스트)":
    # Manual 모드일 때만 위젯을 렌더링하고 그 값을 최종 상태에 할당합니다.
    st.sidebar.info("수동 모드: 아래 설정으로 UI가 고정됩니다.")
    manual_emotion = st.sidebar.selectbox("감정 테스트", config.EMOTIONS)
    manual_accident = st.sidebar.checkbox("사고 발생 테스트")

    st.session_state.current_emotion = manual_emotion
    st.session_state.is_accident_state = manual_accident

elif mode == "Auto (센서 연동)":
    # Auto 모드일 때만 시뮬레이터 연동 로직을 수행합니다.
    st.sidebar.success("🟢 센서/시뮬레이터 연결됨")

    # 자동 새로고침 체크박스 (Session State와 연결)
    st.session_state.auto_refresh_enabled = st.sidebar.checkbox(
        "실시간 데이터 수신 (1초)",
        value=st.session_state.auto_refresh_enabled,
        key="auto_refresh_checkbox"
    )

    # 파일에서 데이터 로드
    detected_emotion, is_accident_detected = load_state()

    new_emotion = detected_emotion
    new_accident = is_accident_detected

    # 값 결정 (우선순위: 사고 > 졸음/closed > 기타 감정)
    if new_accident:
        new_emotion = "panic"  # 사고 시 무조건 panic
        new_accident = True
    elif new_emotion == "closed":
        pass  # closed 유지
    else:
        pass  # 기타 감정 유지

    st.session_state.current_emotion = new_emotion
    st.session_state.is_accident_state = new_accident

    st.sidebar.write(f"감지 감정: {detected_emotion}")
    st.sidebar.write(f"사고 여부: {is_accident_detected}")

    # 자동 새로고침 루프
    if st.session_state.auto_refresh_enabled:
        time.sleep(1)
        st.rerun()

# 3. UI 레이아웃 (최종 결정된 current_emotion과 is_accident 사용)
col1, col2 = st.columns(2)

# [왼쪽] 외부 디스플레이
with col1:
    st.subheader("차량 외부 디스플레이")
    # is_accident 대신 st.session_state.is_accident_state 사용
    if st.session_state.is_accident_state:
        render_external_display("❗ 사고 발생", "#ff4444")
    # current_emotion 대신 st.session_state.current_emotion 사용
    elif st.session_state.current_emotion == "panic":
        render_external_display("⚠️ 급정지 주의", "#ff4444")
    elif st.session_state.current_emotion == "closed":
        render_external_display("😴 졸음 운전", "#ffbb33")
    # ... (나머지 감정 로직도 모두 st.session_state.current_emotion으로 변경) ...
    elif st.session_state.current_emotion == "anger":
        render_external_display("😡 배려 운전", "#ff8800")
    elif st.session_state.current_emotion == "sadness":
        render_external_display("💧 안전 운전", "#ffaa00")
    else:
        render_external_display("😊 안전 운전", "#44dd88")

# [오른쪽] 실내 AI 비서 (차두리)
with col2:
    st.subheader("실내 AI 동작")

    if st.session_state.is_accident_state:
        st.error("💥 충격이 감지되었습니다!")
        render_verify_emergency_ui()
    else:
        current_emotion_state = st.session_state.current_emotion
        # 게임 중이면 게임 유지, 아니면 감정에 따라 행동
        if st.session_state.word_game_active:
            effective_action = "ASK_WORD_GAME"
        else:
            # LLM 판단 로직 (감정 매핑)
            if current_emotion_state == "closed":
                effective_action = "ASK_WORD_GAME"
            elif current_emotion_state == "anger":
                effective_action = "GUIDE_MEDITATION"
            elif current_emotion_state == "happy":
                effective_action = "ASK_CALM_MUSIC"
            elif current_emotion_state == "sadness":
                effective_action = "ASK_UPBEAT_MUSIC"
            elif current_emotion_state == "panic":
                effective_action = "VERIFY_EMERGENCY"
            else:
                effective_action = "NONE"

        # 행동별 UI 렌더링
        if effective_action == "GUIDE_MEDITATION":
            render_meditation_ui()
        elif effective_action == "ASK_WORD_GAME":
            # 게임 시작 트리거
            if not st.session_state.word_game_active and current_emotion == "closed":
                if st.button("졸려 보이시네요. 끝말잇기 할까요?"):
                    st.session_state.word_game_active = True
                    st.rerun()
            if st.session_state.word_game_active:
                render_word_game_ui()
        elif effective_action == "ASK_CALM_MUSIC":
            render_music_ui(calm_music_list, "차분한 음악 추천")
        elif effective_action == "ASK_UPBEAT_MUSIC":
            render_music_ui(upbeat_music_list, "신나는 음악 추천")
        elif effective_action == "VERIFY_EMERGENCY":
            render_verify_emergency_ui()
        else:
            st.info("안전 운전 중입니다. 쾌적한 주행 되세요! 🚗")
