import streamlit as st
import os
from google import genai
from google.genai import types

# --- 1. 配置與核心 AI 邏輯 ---

# 嘗試從 Streamlit Secrets 或環境變數讀取 API Key
api_key = None
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
elif os.environ.get('GEMINI_API_KEY'):
    api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    st.error("找不到 GEMINI_API_KEY。請檢查 Streamlit Secrets 或環境變數。")
    st.stop()


# 設定系統提示基礎 (基礎角色與行為規範)
BASE_SYSTEM_INSTRUCTION = (
    "你是一位幽默詼諧的諷刺大師。你的任務是根據家長訊息和「老師當前的情緒」，生成一則供老師內部觀賞、用於情緒發洩的幽默回覆。"
    "回覆必須**不能**發送給家長，目的是讓老師感到舒壓。"
    "你的回覆原則如下：語氣風格偏冷面幽默、諷刺、一本正經地講荒謬的話，不使用正式公文語氣，不討好、不安撫、不道歉。"
    "幽默策略：優先使用誇張、反問、邏輯拆解來凸顯對方的不合理，可以「假裝很認真」地順著家長邏輯講到荒謬的結論。允許輕度嘲諷，但不使用人身攻擊或髒話。"
    "字數請盡量控制在 150 個中文字元以內。"
)

@st.cache_resource
def get_gemini_client():
    """快取 Gemini Client，避免重複初始化。"""
    # 確保 client 能夠成功初始化
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"初始化 Gemini 失敗：{e}")
        st.stop()


def generate_dinosaur_parent_response(parent_message: str, teacher_emotion: str) -> str:
    """
    呼叫 Gemini API，根據家長訊息和老師的憤怒值生成幽默回覆。
    
    Args:
        parent_message: 家長輸入的文字。
        teacher_emotion: 老師選擇的情緒。
    Returns:
        AI 生成的幽默回覆。
    """

    # 1. 根據老師的情緒，動態調整諷刺風格
    style_instruction = ""
    if "怒火中燒" in teacher_emotion:
        style_instruction = "請使用**最強烈、最戲劇化、最歇斯底里**的黑色幽默語氣進行尖銳諷刺。讓回覆充滿爆發性的情緒反差，達到最強的舒壓效果。"
    elif "精疲力盡" in teacher_emotion:
        style_instruction = "請使用**無力、躺平、淡漠**的語氣進行吐槽。諷刺風格要輕描淡寫，但句句紮心，體現老師的無奈感。"
    elif "幽默輕鬆" in teacher_emotion:
        style_instruction = "請使用**溫和、輕鬆**的語氣進行幽默回應，諷刺點到為止，讓回覆看起來有趣但沒有攻擊性。"
    elif "滿頭問號" in teacher_emotion:
        style_instruction = "請使用**極度理性、過度嚴謹**的學術語氣來反駁家長的要求，用文縐縿的語氣將問題的荒謬性放大。"
    
    # 2. 組合最終的 System Instruction
    SYSTEM_INSTRUCTION_FINAL = f"{BASE_SYSTEM_INSTRUCTION}\n\n【本次回覆風格指示】：{style_instruction}"

    # 使用 with 語句創建一個 Spinner
    with st.spinner(f"⏳ AI 老師正在感知您的情緒 ({teacher_emotion})，並構思回覆中..."):
        client = get_gemini_client()
        
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_FINAL, # 使用動態指令
            temperature=0.7, # 稍微調高溫度以增加幽默感
        )

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=parent_message,
                config=config,
            )
            # Spinner 會在 with 區塊結束後自動消失
            return response.text
            
        except Exception as e:
            st.error(f"❌ 處理失敗：{e}")
            return "很抱歉，系統目前無法處理您的請求。"


# --- 2. Streamlit 網頁界面 ---

st.set_page_config(page_title="🦖 恐龍家長專業回覆機", layout="wide")

# 確保所有 st.session_state 變數在使用前都被定義
if 'ai_reply' not in st.session_state:
    st.session_state.ai_reply = "尚未收到任何回覆，請在上方輸入家長訊息並點擊送出。"


st.title("🦖 恐龍家長專業回覆機 (Gemini AI)")

# 老師情緒選擇區
teacher_emotion = st.selectbox(
    "💬 老師您看完後的情緒是？",
    ["😠 怒火中燒", "🤣 幽默輕鬆", "😫 精疲力盡", "🤨 滿頭問號"]
)
st.caption(f"（AI 將根據您選擇的 **{teacher_emotion.split(' ')[0]}** 情緒，調整回覆的諷刺強度）")


# 恐龍家長輸入區
parent_message = st.text_area(
    "請輸入恐龍家長訊息：", 
    height=150, 
    placeholder="例如：老師，我兒子說他功課已經寫完了，你們一定要他檢查三次是在浪費時間！請取消這個規定！"
)


# 送出按鈕
if st.button("送出訊息給 AI 老師"):
    if parent_message:
        # 呼叫 AI 核心邏輯，傳遞兩個參數
        ai_response = generate_dinosaur_parent_response(parent_message, teacher_emotion)
        st.session_state.ai_reply = ai_response
        st.rerun() # 觸發 rerun 以立即顯示結果
    else:
        st.error("請輸入訊息！")


# AI 老師回覆區
st.markdown("---")
st.subheader("AI 老師的（內部舒壓用）回覆：")
st.info(st.session_state.ai_reply)

# 底部說明
st.caption("本工具目的為教師情緒抒發，回覆內容幽默、諷刺，請勿將其用於正式對外溝通。")
st.caption("本專題應用程式使用 Streamlit 和 Google Gemini API 串接。")