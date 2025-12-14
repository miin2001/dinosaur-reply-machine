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
    "你不是客服、不是老師、也不是學校行政人員。你是一個「毒舌的喜劇演員」或「滿腹經綸的諷刺大師」，專門替人撰寫回覆恐龍家長的文字。"
    "你的回覆原則如下：語氣風格風格偏冷面幽默、諷刺、一本正經地講荒謬的話，不使用正式公文語氣，不討好、不安撫、不道歉，除非道歉本身是反諷"
    "幽默策略如下：優先使用誇張、反問、邏輯拆解來凸顯對方的不合理，可以「假裝很認真」地順著家長邏輯講到荒謬的結論。允許輕度嘲諷，但不使用人身攻擊或髒話"
    "字數大約100個中文字元"
)

@st.cache_resource
def get_gemini_client():
    """快取 Gemini Client，避免重複初始化。"""
    return genai.Client(api_key=api_key)


def generate_dinosaur_parent_response(parent_message: str) -> str:
    """呼叫 Gemini API，根據家長訊息生成專業回覆。"""
    #st.session_state.status_message = "⏳ 正在將家長訊息送給 AI 老師處理..."
    #st.rerun() # 重新運行以更新狀態顯示

    # 使用 with 語句創建一個 Spinner
    with st.spinner("⏳ AI 老師正在溫和地構思回覆..."):
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
            # Spinner 會在 with 區塊結束後自動消失
            return response.text
            
        except Exception as e:
            st.error(f"❌ 處理失敗：{e}")
            return "很抱歉，系統目前無法處理您的請求。"

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

# 確保所有 st.session_state 變數在使用前都被定義
if 'ai_reply' not in st.session_state:
    st.session_state.ai_reply = "尚未收到任何回覆，請在上方輸入家長訊息並點擊送出。"


# 初始化 Session State 來儲存狀態和回覆
#if 'ai_reply' not in st.session_state:
#   st.session_state.ai_reply = "等待家長訊息中..."
#if 'status_message' not in st.session_state:
#    st.session_state.status_message = "準備就緒..."

st.title("🦖 恐龍家長專業回覆機 (Gemini AI)")

# 顯示狀態
#st.markdown(f"**狀態:** <span style='color: #007bff; font-weight: bold;'>{st.session_state.status_message}</span>", unsafe_allow_html=True)

# 恐龍家長輸入區
parent_message = st.text_area(
    "請輸入恐龍家長訊息：", 
    height=150, 
    placeholder="例如：老師，我兒子說他功課已經寫完了，你們一定要他檢查三次是在浪費時間！請取消這個規定！"
)

# 語音輸入的替代方案
#st.markdown("---")
st.caption("我們將專注於文字輸入和 AI 邏輯。")


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