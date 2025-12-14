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
    st.stop() # 確保在沒有 key 時停止運行

@st.cache_resource
def get_gemini_client():
    """快取 Gemini Client，避免重複初始化。"""
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        # 這裡不應該 st.error 和 st.stop，讓上層呼叫處理
        raise RuntimeError(f"初始化 Gemini 失敗：{e}")


# 設定系統提示基礎 (基礎角色與行為規範)
SYSTEM_INSTRUCTION_HUMOR = (
    "你是一位極度毒舌且擁有黑色幽默的諷刺大師。你的任務是為老師生成一段用於**情緒發洩**的內部回覆。"
    "回覆風格：冷面幽默、諷刺、一本正經地講荒謬的話，目的讓老師感到舒壓。"
    "幽默策略：使用誇張、反問、邏輯拆解，不使用正式公文語氣，字數控制在 150 字元以內。"
)

def analyze_emotion(message: str) -> str:
    """
    呼叫 Gemini API，專門判斷訊息中的核心情緒。
    回傳範例: "憤怒|要求"
    """
    client = get_gemini_client()
    
    # 嚴格的提示詞，要求模型只輸出關鍵情緒詞
    emotion_prompt = (
        "請仔細分析以下家長訊息，判斷其中最強烈且最相關的情緒和意圖。 "
        "你只能從以下選項中選擇一個或多個，並用 | 符號連接，不允許任何額外解釋和前綴。\n"
        "選項: [憤怒, 焦慮, 不滿, 質疑, 無助, 要求, 抱怨, 平靜, 感謝]\n"
        "家長訊息:\n"
        f"---{message}---"
    )
    
    config = types.GenerateContentConfig(temperature=0.1) # 溫度設低，要求精確性
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=emotion_prompt,
            config=config
        )
        # 清理並回傳結果
        return response.text.strip().replace('"', '').replace("'", "")
        
    except Exception as e:
        return f"分析失敗: {e}"


def generate_dinosaur_parent_response(parent_message: str) -> str:
    """呼叫 Gemini API，生成幽默回覆。"""
    
    client = get_gemini_client()
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_HUMOR,
        temperature=0.6,
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parent_message,
            config=config,
        )
        return response.text
        
    except Exception as e:
        # 在這裡捕獲錯誤並顯示
        st.error(f"❌ 生成回覆失敗：{e}")
        return "很抱歉，系統目前無法處理您的請求。"


# --- 2. Streamlit 網頁界面 ---

st.set_page_config(page_title="🦖 恐龍家長專業回覆機", layout="wide")

# 確保所有 st.session_state 變數在使用前都被定義
if 'ai_reply' not in st.session_state:
    st.session_state.ai_reply = "尚未收到任何回覆，請在上方輸入家長訊息並點擊送出。"
if 'parent_emotion' not in st.session_state:
    st.session_state.parent_emotion = "未分析"
    st.session_state.emotion_icon = "❓"

st.title("🦖 恐龍家長情緒分析與舒壓回覆機 (Gemini AI)")
st.markdown("---")


# 恐龍家長輸入區
parent_message = st.text_area(
    "請輸入恐龍家長訊息：", 
    height=150, 
    placeholder="例如：老師，我兒子說他功課已經寫完了，你們一定要他檢查三次是在浪費時間！請取消這個規定！"
)

# 送出按鈕
if st.button("送出訊息給 AI 老師"):
    if parent_message:
        # 1. 情緒分析步驟
        with st.spinner("🧠 正在進行情緒分析..."):
            emotion_result = analyze_emotion(parent_message)
            st.session_state.parent_emotion = emotion_result
        
        # 2. 呼叫幽默回覆生成
        with st.spinner("⏳ 正在生成幽默諷刺回覆..."):
            ai_response = generate_dinosaur_parent_response(parent_message)
            st.session_state.ai_reply = ai_response
            
        st.rerun() # 觸發 rerun 立即更新所有狀態和顯示結果
    else:
        st.error("請輸入訊息！")


# --- 3. 結果顯示區 ---

st.markdown("---")

col1, col2 = st.columns([1, 2])

# 情緒分析結果顯示 (左側)
with col1:
    st.subheader("家長情緒分析")
    
    emotion_map = {
        "憤怒": "🔴 怒火中燒", 
        "焦慮": "🟠 擔憂不安", 
        "要求": "🟡 強勢要求", 
        "不滿": "🔵 不吐不快",
        "無助": "🟣 束手無策",
        "平靜": "🟢 理性溝通",
        "感謝": "⭐ 感謝讚許",
        "質疑": "❓ 提出質疑",
        "抱怨": "💢 情緒性抱怨"
    }

    # 處理多個情緒或未分析
    emotions = st.session_state.parent_emotion.split('|')
    display_emotion_text = " / ".join([emotion_map.get(e.strip(), e.strip()) for e in emotions])
    
    if st.session_state.parent_emotion == "未分析":
        st.info("請輸入訊息並點擊送出。")
    elif "分析失敗" in st.session_state.parent_emotion:
        st.warning("情緒分析失敗。")
    else:
        st.metric(
            label="偵測到的主要情緒", 
            value=display_emotion_text
        )

# AI 老師回覆區 (右側)
with col2:
    st.subheader("AI 老師的（內部舒壓用）回覆：")
    st.info(st.session_state.ai_reply)

# 底部說明
st.markdown("---")
st.caption("本工具目的為教師情緒抒發及分析，回覆內容幽默、諷刺，請勿將其用於正式對外溝通。")
st.caption("本專題應用程式使用 Streamlit 和 Google Gemini API 串接。")