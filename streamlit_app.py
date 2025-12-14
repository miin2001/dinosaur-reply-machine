import streamlit as st
import os
from google import genai
from google.genai import types

# --- 1. 配置與核心 AI 邏輯 ---

# Streamlit 處理 API Key 的標準方式：從 Streamlit Secrets 讀取
# 為了本地測試，它也會檢查環境變數
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
elif os.environ.get('GEMINI_API_KEY'):
    api_key = os.environ.get('GEMINI_API_KEY')
else:
    st.error("找不到 GEMINI_API_KEY。請檢查 Streamlit Secrets 或環境變數。")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"初始化 Gemini 失敗：{e}")
    st.stop()

# 設定系統提示 (System Instruction)
SYSTEM_INSTRUCTION = (
    "你是一位資深、專業且經驗豐富的學校老師/行政人員。你正在回覆一位態度強硬、要求不合理的「恐龍家長」的訊息或抱怨。 "
    "你的回覆必須保持**極度禮貌、專業、耐心**，但同時要**堅守學校的教育原則和規定**。 "
    "請溫和地**拒絕**任何違反專業常規或過度干涉的要求，並以教育專家的角度提供**建設性的、符合專業倫理的建議**。"
    "回覆請以中文為主，語氣要堅定但婉轉。"
)

@st.cache_resource
def get_gemini_client():
    """快取 Gemini Client，避免重複初始化。"""
    return genai.Client(api_key=api_key)


def generate_dinosaur_parent_response(parent_message: str) -> str:
    """呼叫 Gemini API，根據家長訊息生成專業回覆。"""
    st.session_state.status_message = "⏳ 正在將家長訊息送給 AI 老師處理..."
    st.experimental_rerun() # 重新運行以更新狀態顯示

    client = get_gemini_client()
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.6,
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parent_message,
            config=config,
        )
        st.session_state.status_message = "✨ AI 老師回覆完成！"
        return response.text
        
    except Exception as e:
        error_message = f"呼叫 Gemini API 發生錯誤: {e}"
        st.session_state.status_message = f"❌ 處理失敗：{e}"
        st.error(error_message)
        return "很抱歉，系統目前無法處理您的請求，請稍後再試。 (請檢查終端機的錯誤訊息)"


# --- 2. Streamlit 網頁界面 ---

st.set_page_config(page_title="🦖 恐龍家長專業回覆機", layout="wide")

# 初始化 Session State 來儲存狀態和回覆
if 'ai_reply' not in st.session_state:
    st.session_state.ai_reply = "等待家長訊息中..."
if 'status_message' not in st.session_state:
    st.session_state.status_message = "準備就緒..."

st.title("🦖 恐龍家長專業回覆機 (Gemini AI)")

# 顯示狀態
st.markdown(f"**狀態:** <span style='color: #007bff; font-weight: bold;'>{st.session_state.status_message}</span>", unsafe_allow_html=True)

# 恐龍家長輸入區
parent_message = st.text_area(
    "請輸入恐龍家長訊息：", 
    height=150, 
    placeholder="例如：老師，我兒子說他功課已經寫完了，你們一定要他檢查三次是在浪費時間！請取消這個規定！"
)

# 語音輸入的替代方案
st.markdown("---")
st.warning("⚠️ Streamlit 本身不直接支援瀏覽器語音輸入。若需語音，建議使用 Streamlit 擴充元件或電腦系統的語音轉文字功能。")
st.caption("為簡化專題，我們將專注於文字輸入和 AI 邏輯。")


# 送出按鈕
if st.button("送出訊息給 AI 老師"):
    if parent_message:
        # 呼叫 AI 核心邏輯
        ai_response = generate_dinosaur_parent_response(parent_message)
        st.session_state.ai_reply = ai_response
    else:
        st.session_state.status_message = "請輸入訊息！"


# AI 老師回覆區
st.markdown("---")
st.subheader("AI 老師回覆：")
st.info(st.session_state.ai_reply)

# 底部說明
st.caption("本專題應用程式使用 Streamlit 和 Google Gemini API 串接。")