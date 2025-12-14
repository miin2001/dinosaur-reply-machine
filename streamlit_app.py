import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import json
from google import genai
from google.genai import types
import matplotlib.font_manager as fm

# --- I. 常量和輔助函式定義 (保持核心邏輯) ---

# 常用代表性顏色（可以自行擴增）
COLOR_NAMES = {
    "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
    "red": (220, 20, 60), "orange": (255, 140, 0), "yellow": (255, 215, 0),
    "green": (34, 139, 34), "blue": (30, 144, 255), "purple": (147, 112, 219),
    "pink": (255, 182, 193), "brown": (139, 69, 19),
    "beige": (245, 245, 220), "navy": (0, 0, 128), "olive": (85, 107, 47)
}

def rgb_to_hex(rgb):
    """輔助函式：將 RGB 轉換為 Hex 碼"""
    return '#%02x%02x%02x' % tuple(rgb)

def closest_color_name(rgb):
    """計算最接近的定義顏色名稱"""
    r, g, b = rgb
    min_dist = float("inf")
    closest_name = None
    for name, value in COLOR_NAMES.items():
        vr, vg, vb = value
        dist = (r - vr)**2 + (g - vg)**2 + (b - vb)**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def color_style_tags(rgb):
    """根據色彩學屬性判斷風格標籤 (使用優化邏輯)"""
    r, g, b = rgb
    tags = []
    
    brightness = (r + g + b) / 3
    chroma = max(r, g, b) - min(r, g, b)

    # 冷暖色
    if b > r and b > g:
        tags.append("cool")
    elif r > b and g > b:
         tags.append("warm")
    else: 
         tags.append("neutral") 

    # 明亮 vs 暗色
    if brightness > 200:
        tags.append("bright")
    elif brightness < 60:
        tags.append("dark")
        
    # 鮮豔 vs 低彩度
    if chroma > 100:
         tags.append("vivid")
    elif chroma < 30:
         tags.append("muted")

    # 高級感：低彩度深色
    if brightness < 120 and chroma < 50:
        tags.append("luxury")

    # 自然色
    if (r > 120 and g > 100 and b < 80 and chroma > 20):
         tags.append("natural") 

    return tags

def find_chinese_font():
    """嘗試自動尋找常用的中文字體路徑"""
    common_fonts = [
        '/System/Library/Fonts/PingFang.ttc',           # macOS
        '/System/Library/Fonts/STHeiti Light.ttc',      # macOS
        'C:/Windows/Fonts/msjh.ttc',                    # Windows (微軟正黑體)
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # Linux (文泉驛)
    ]
    for font_path in common_fonts:
        if os.path.exists(font_path):
            return font_path
    return None

def setup_chinese_font():
    """設定 Matplotlib 使用的中文字體"""
    font_path = find_chinese_font()
    if font_path:
        zh_font = fm.FontProperties(fname=font_path, size=10)
        plt.rcParams['font.family'] = zh_font.get_name()
    else:
        # 如果找不到，設定回退字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False # 解決負號亂碼

# --- II. 核心功能函式 ---

# 避免每次運行時都重新執行 K-means，使用 st.cache_data 提高效率
@st.cache_data
def extract_colors(image, k=5):
    """K-means 顏色提取 (加入 random_state 和亮度排序)"""
    img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)
    
    # 按亮度排序 (Luminance)
    # L = 0.299*R + 0.587*G + 0.114*B (常用公式)
    luminances = np.dot(colors, [0.299, 0.587, 0.114]) 
    sorted_indices = np.argsort(luminances)
    
    return colors[sorted_indices]

# 避免每次運行時都重新呼叫 Gemini API
@st.cache_data
def generate_brand_moodboard_content(color_data, api_key):
    """
    使用 Gemini API 根據顏色數據生成品牌關鍵字和氛圍描述。
    """
    # 初始化 Client
    try:
        # Key 從參數傳入
        client = genai.Client(api_key=api_key) 
    except Exception as e:
        # 在 Streamlit 應用中，最好拋出更清晰的錯誤
        raise ConnectionError(f"API Client 初始化失敗，Key 無效或網路錯誤: {e}")
        
    color_info_list = []
    for rgb, name, tags in color_data:
        color_info_list.append({
            "hex": rgb_to_hex(rgb),
            "name": name.capitalize(),
            "style_tags": tags
        })
        
    # Prompt 設計
    color_input_str = json.dumps(color_info_list, ensure_ascii=False, indent=2)
    
    prompt = f"""
    你是一位頂尖的品牌策略顧問和色彩心理學專家。
    請根據以下的色票資訊，為一個新品牌生成一份品牌形象的草稿。

    色票數據：
    {color_input_str}

    請生成以下內容，並**嚴格以 Markdown 格式的 JSON 區塊**輸出。
    
    1. **Brand_Keywords (列表, 5-7個)**：根據整體色調帶來的聯想，列出品牌核心關鍵字 (e.g., 奢華, 自然, 科技, 溫暖)。
    2. **Brand_Vibe_Description (字串, 150字以內)**：綜合所有顏色，寫一段精煉的品牌氛圍描述，說明品牌給人的整體感受和情感連結。
    3. **Color_Analysis (列表)**：針對**每一個**色票，生成一段簡短的分析 (約30字)，說明該顏色在品牌中的作用和象徵意義。

    輸出格式範例:
    ```json
    {{
      "Brand_Keywords": ["...", "..."],
      "Brand_Vibe_Description": "...",
      "Color_Analysis": [
        {{ "hex": "#...", "analysis": "..." }},
        {{ "hex": "#...", "analysis": "..." }}
      ]
    }}
    ```
    """

    # 呼叫 Gemini API
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt
    )
    
    json_text = response.text.strip()
    if json_text.startswith("```json"):
        json_text = json_text.lstrip("```json").rstrip("```").strip()
        
    return json.loads(json_text)

# --- III. Matplotlib 繪圖函式 (專為 Streamlit 優化) ---

def create_palette_figure(colors):
    """
    創建一個只包含色票、Hex 碼和名稱的 Matplotlib Figure。
    """
    # 建立 Matplotlib 圖表
    fig, ax = plt.subplots(figsize=(10, 1.8))
    
    palette_height = 50
    width_per_color = 100 # 增加色塊寬度
    palette = np.zeros((palette_height, colors.shape[0] * width_per_color, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        start_x = i * width_per_color
        end_x = (i+1) * width_per_color
        palette[:, start_x:end_x] = color
        
        hex_code = rgb_to_hex(color)
        color_name = closest_color_name(color).capitalize()
        
        # 顯示 Hex 碼
        ax.text(start_x + width_per_color/2, palette_height + 5, hex_code, 
                 ha='center', va='top', fontsize=10, color='black')
        # 顯示顏色名稱
        ax.text(start_x + width_per_color/2, palette_height + 20, color_name, 
                 ha='center', va='top', fontsize=10, color='black')

    ax.imshow(palette)
    ax.set_title("K-means 萃取色票 (Color Palette)", fontsize=12)
    ax.axis('off')
    ax.set_ylim(palette_height + 40, 0) # 調整 y 軸範圍以容納文字
    
    plt.tight_layout()
    return fig

# --- IV. Streamlit 應用程式主體 ---

def main():
    # 設置中文字體 (在應用程式開始時執行一次)
    setup_chinese_font()
    
    # Streamlit 網頁設定
    st.set_page_config(
        page_title="圖片情緒板生成器 (Image Moodboard Generator)", 
        layout="wide"
    )
    
    st.title("🎨 AI 圖片情緒板與品牌風格生成器")
    st.markdown("上傳一張圖片，利用 K-means 提取核心色票，並透過 Gemini AI 生成moodboard。")

    # --- 關鍵變動：從 st.secrets 讀取 API Key ---
    try:
        # 從 Streamlit Cloud 的 Secrets 中讀取 Key
        # 假設您的 Key 名稱是 GEMINI_API_KEY
        api_key = st.secrets["GEMINI_API_KEY"] 
    except Exception:
        # 如果 Key 沒設置，給予警告
        st.error("❌ Gemini API Key 未在 Streamlit Secrets 中設定！請檢查 `.streamlit/secrets.toml` 文件或 Streamlit Cloud 應用程式設定。")
        return
    # --- 關鍵變動結束 ---

    # 側邊欄輸入區 (只保留 K 值和圖片上傳)
    with st.sidebar:
        st.header("參數與設定")
        
        # 圖片上傳區
        uploaded_file = st.file_uploader(
            "選擇一張圖片 (.jpg, .png)", 
            type=["jpg", "jpeg", "png"]
        )
        
        # K 值選擇
        k_clusters = st.slider("選擇色票數量 (K 值)", 3, 10, 5, 1)

    # 主內容區塊
    
    if uploaded_file is None:
        st.info("請在側邊欄上傳圖片以開始分析。")
        return

    # 1. 讀取與顯示圖像
    try:
        # 讀取上傳的圖片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"圖像讀取失敗: {e}")
        return

    # 使用 Streamlit 欄位佈局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("1. 原始輸入")
        st.image(img_rgb, caption=uploaded_file.name, use_column_width=True)

    # 2. 執行分析
    with col2:
        st.header("2. 色彩與風格分析")
        
        # K-means 顏色提取
        with st.spinner(f'正在進行 K-means 顏色提取 (K={k_clusters})...'):
            # Streamlit 的 @st.cache_data 會幫助優化性能
            colors = extract_colors(img_rgb, k=k_clusters) 
            
        # 整理 LLM 輸入資料
        final_color_data = []
        for c in colors:
            name = closest_color_name(c)
            tags = color_style_tags(c)
            final_color_data.append((c, name, tags))
            
        # 顯示色票圖
        st.subheader("萃取色票面板")
        fig_palette = create_palette_figure(colors)
        st.pyplot(fig_palette, use_container_width=True)

        # --- 新增內容：顯示風格標籤（作為中介數據展示） ---
        st.subheader("中介數據：色彩意象標籤")
        
        # 創建一個可展開的區塊，用於技術展示
        with st.expander("點擊查看 K-means 顏色提取的原始風格標籤 (供 AI 參考)"):
            tag_data = []
            for c, name, tags in final_color_data:
                tag_data.append({
                    "Hex Code": rgb_to_hex(c),
                    "名稱": name.capitalize(),
                    "風格標籤 (Tags)": ", ".join(tags)
                })
            
            # 使用 DataFrame 顯示
            st.dataframe(tag_data, hide_index=True)

        # --- 新增內容結束 ---
        
        # Gemini AI 生成
        with st.spinner('🎨 正在呼叫 Gemini AI 生成品牌氛圍描述...'):
            try:
                # 關鍵變動：將 api_key 變數傳給函式
                llm_result = generate_brand_moodboard_content(final_color_data, api_key) 
            except ConnectionError as ce:
                # 顯示錯誤，但程式碼不會暴露 Key
                st.error(f"Gemini AI 呼叫失敗。請確認 Streamlit Secrets 中的 Key 是否有效且網路連線正常。錯誤詳情: {ce}") 
                return
            except Exception as e:
                st.error(f"Gemini AI 生成內容失敗，可能是 AI 輸出格式錯誤或 Key 權限問題。錯誤: {e}")
                return

    st.markdown("---")
    
    # 3. 顯示 Gemini 生成的文字內容
    # 3. 顯示 Gemini 生成的文字內容
    if llm_result:
        st.header("3. 品牌風格 Moodboard 內容")
        
        keywords = "｜".join(llm_result.get("Brand_Keywords", ["無關鍵字"]))
        vibe_desc = llm_result.get("Brand_Vibe_Description", "無描述")
        
        st.markdown(f"**核心關鍵字 (Keywords):** **`{keywords}`**")
        st.info(vibe_desc) # 用 info 框顯示氛圍描述，視覺上更突出

        st.subheader("詳細顏色分析 (Color Analysis)")
        
        analysis_items = llm_result.get("Color_Analysis", [])

        if analysis_items:
            
            # --- 使用 HTML/Markdown 建立自定義表格以顯示色塊 ---
            
            # 使用更簡潔的 CSS 塊，並確保它在整個 HTML 結構的頂部
            html_content = """
            <style>
                .color-block {
                    width: 30px; /* 色塊寬度 */
                    height: 30px; /* 色塊高度 */
                    border: 1px solid #ccc; /* 邊框 */
                    border-radius: 4px; /* 圓角 */
                    display: inline-block; /* 行內區塊 */
                    vertical-align: middle; /* 垂直居中 */
                }
                .analysis-table {
                    width: 100%;
                    border-collapse: collapse; /* 消除邊框間隙 */
                    margin-top: 15px;
                }
                .analysis-table th, .analysis-table td {
                    padding: 12px 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .analysis-table th {
                    background-color: #f0f2f6; /* Streamlit 淺灰色背景 */
                }
            </style>
            
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th style="width: 10%;">色票</th>
                        <th style="width: 20%;">Hex Code</th>
                        <th style="width: 70%;">分析與作用 (Gemini Analysis)</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # 迭代生成每一行內容
            for item in analysis_items:
                hex_code = item.get("hex", "#FFFFFF")
                analysis = item.get("analysis", "AI 未提供分析內容")
                
                # 創建色塊的 HTML 元素
                color_block_html = f'<div class="color-block" style="background-color: {hex_code};"></div>'
                
                html_content += f"""
                <tr>
                    <td>{color_block_html}</td>
                    <td><code>{hex_code}</code></td>
                    <td>{analysis}</td>
                </tr>
                """
            
            # 關閉表格標籤
            html_content += "</tbody></table>"
            
            # 渲染整個 HTML 內容
            st.markdown(html_content, unsafe_allow_html=True)
            
        else:
            st.warning("AI 未能提供顏色分析內容。")

if __name__ == '__main__':
    main()