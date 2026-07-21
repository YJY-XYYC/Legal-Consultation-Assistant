import streamlit as st
import streamlit.components.v1 as components
import os
import subprocess
import sys
import http.server
import socketserver
import threading
import time

# 修复导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必须在所有 huggingface 相关导入之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

# -------------------------- 
# 导入模块
# --------------------------
try:
    from modules.rag_module import RAGModule
    from modules.agent_module import AgentModule
    from config import validate_config, create_directories
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.stop()

# --------------------------
# 页面配置
# --------------------------
st.set_page_config(
    page_title="劳动与社会保障法律法规知识智能助手",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚖️"
)

# --------------------------
# 自定义样式
# --------------------------
st.markdown("""
<style>
    .main-header {
        position: fixed !important;
        top: 0 !important;
        left: 21rem !important;
        right: 0 !important;
        height: 3.5rem !important;
        line-height: 3.5rem !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #1E3A5F !important;
        text-align: center !important;
        padding: 0 220px 0 60px !important;
        margin: 0 !important;
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: none !important;
        box-shadow: none !important;
        z-index: 2147483600 !important;
        transition: left 0.3s ease, right 0.3s ease, background 0.3s ease, color 0.3s ease !important;
        box-sizing: border-box !important;
        pointer-events: none !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .dark .main-header {
        color: #E3F2FD !important;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
        color: #1E3A5F;
    }
    .dark .info-box {
        background-color: #1E3A5F;
        border-left: 4px solid #64B5F6;
        color: #E3F2FD;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 4px solid #FF9800;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .btn-primary {
        background-color: #1E3A5F;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    .thinking-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: transparent;
        border-left: 4px solid #2196F3;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    /* ===== 底部固定输入区样式（JS 内联样式为主，CSS 作为 fallback） ===== */
    .st-key-fixed_input {
        position: fixed !important;
        bottom: 0 !important;
        left: 21rem !important;
        width: calc(100vw - 21rem) !important;
        max-width: calc(100vw - 21rem) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,1) 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 14px 24px 18px 24px !important;
        z-index: 999 !important;
        transition: left 0.3s ease, width 0.3s ease, max-width 0.3s ease !important;
        box-sizing: border-box !important;
    }
    /* 给主内容区添加底部内边距，避免内容被固定输入区遮挡 */
    .main .block-container {
        padding-bottom: 240px !important;
        max-height: calc(100vh - 240px) !important;
        overflow-y: auto !important;
    }
    /* 隐藏 container 内部多余 padding */
    .st-key-fixed_input > div,
    .st-key-fixed_input > div {
        padding: 0 !important;
    }
    /* 让 Streamlit 原生 chat_input 透明融入固定区 */
    .stChatInput {
        padding: 0 !important;
        background: transparent !important;
    }
    .stChatInput > div {
        background: transparent !important;
        border-radius: 24px !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: none !important;
    }
    /* 注：不再额外改 textarea/button 的 flex 行为，保留 BaseWeb 原生布局
       以免 min-width:0 等规则破坏输入框的尺寸计算。 */
    /* 固定输入区内的 textarea - 浅色模式白色背景 */
    .st-key-fixed_input textarea[data-testid="stChatInputTextArea"],
    .st-key-fixed_input textarea[data-testid="stChatInputTextArea"] {
        background-color: rgb(255, 255, 255) !important;
        color: rgb(49, 51, 63) !important;
    }
    /* 固定输入区内的 textarea - 暗黑模式深色背景 */
    [data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"],
    [data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"],
    .stApp[data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"],
    .stApp[data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"] {
        background-color: rgb(38, 39, 48) !important;
        color: rgb(250, 250, 250) !important;
        caret-color: rgb(250, 250, 250) !important;
    }
    /* placeholder 颜色 - 暗黑模式 */
    [data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"]::placeholder,
    [data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"]::placeholder,
    .stApp[data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"]::placeholder,
    .stApp[data-theme="dark"] .st-key-fixed_input textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: rgba(250, 250, 250, 0.5) !important;
    }
    /* 复选框在固定区内紧凑显示 */
    .st-key-fixed_input .stCheckbox,
    .st-key-fixed_input .stCheckbox {
        margin-bottom: 0 !important;
    }
    /* 固定输入区内下拉菜单 - 透明背景融入 */
    .st-key-fixed_input .stSelectbox {
        padding-top: 12px;
    }
    .st-key-fixed_input .stSelectbox [data-baseweb="select"] > div:first-child {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .st-key-fixed_input .stSelectbox [data-baseweb="select"] > div:first-child > div:first-child {
        background-color: transparent !important;
    }
    /* 暗黑模式下文字颜色 */
    [data-theme="dark"] .st-key-fixed_input .stSelectbox [data-baseweb="select"] [role="combobox"] ~ div,
    .stApp[data-theme="dark"] .st-key-fixed_input .stSelectbox [data-baseweb="select"] [role="combobox"] ~ div {
        color: rgb(250, 250, 250) !important;
        background-color: transparent !important;
    }
    /* 统一两个 Tab 底部控制行高度 */
    .st-key-fixed_input [data-testid="stHorizontalBlock"] [data-testid="stColumn"],
    .st-key-fixed_input [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
        display: flex !important;
        align-items: center !important;
    }
    .st-key-fixed_input .stCheckbox label,
    .st-key-fixed_input .stCheckbox label {
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
    }


    /* 文件上传区域宽度 */
    .st-key-fixed_input [data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        max-width: none !important;
    }
    /* 文件上传 Tab 提交按钮与 dropzone 等高并垂直居中 */
    .st-key-fixed_input [data-testid="stForm"]:has(.stFileUploader) [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
        align-items: stretch !important;
    }
    .st-key-fixed_input [data-testid="stForm"]:has(.stFileUploader) [data-testid="stHorizontalBlock"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    .st-key-fixed_input [data-testid="stForm"]:has(.stFileUploader) .stFormSubmitButton {
        height: 100% !important;
    }
    .st-key-fixed_input [data-testid="stForm"]:has(.stFileUploader) .stFormSubmitButton button {
        min-height: 65px !important;
        display: flex !important;
        align-items: center !important;
    }
    /* ===== 顶部固定导航栏（radio 导航固定到页面顶部） ===== */
    div[role="radiogroup"][aria-label="导航"] {
        position: fixed !important;
        top: 3.5rem !important;
        left: 21rem !important;
        right: 0 !important;
        z-index: 998 !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,1) 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 0 24px !important;
        margin: 0 !important;
        border-bottom: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        transition: left 0.3s ease, right 0.3s ease, background 0.3s ease !important;
        box-sizing: border-box !important;
    }
    /* ===== 全局 emoji 字体栈，确保 emoji 使用系统原生彩色字体渲染 ===== */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
        font-family: "Source Sans Pro", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif;
    }
    /* 保护 Material Icons 字体不被全局规则覆盖，确保 toggle 箭头等图标正常渲染 */
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    /* 导航 radio 按钮 emoji 字体栈，确保 emoji 使用系统原生彩色字体渲染 */
    div[role="radiogroup"][aria-label="导航"],
    div[role="radiogroup"][aria-label="导航"] * {
        font-family: "Source Sans Pro", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif !important;
    }
    /* 导航 radio 按钮内 emoji 不受文字颜色影响，保持原生彩色 */
    div[role="radiogroup"][aria-label="导航"] p {
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", "Source Sans Pro", sans-serif !important;
    }
    /* 导航 radio 选中项背景色变为淡红色 */
    div[role="radiogroup"][aria-label="导航"] label:has(input[type="radio"]:checked) {
        background-color: rgba(255, 0, 0, 0.18) !important;
    }
    /* 导航三个区域样式 */
    div[role="radiogroup"][aria-label="导航"] label {
        padding: 4px 12px !important;
        font-size: 12px !important;
        border-radius: 8px !important;
    }
    /* 隐藏 tab 下方的红色边框指示线 */
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
    /* 隐藏所有 markdown 分割线（hr） */
    div[data-testid="stMarkdownContainer"] hr {
        display: none !important;
    }
    /* 给主内容区增加上内边距，避免内容被固定导航栏遮挡 */
    .main .block-container {
        padding-top: 110px !important;
    }
    /* 聊天消息宽度随内容自适应 */
    [data-testid="stChatMessage"] {
        width: fit-content !important;
        max-width: 100% !important;
        background: transparent !important;
    }
    /* 用户消息气泡 */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p {
        background: #262730 !important;
        padding: 8px 14px !important;
        border-radius: 12px !important;
        display: inline-block !important;
    }

    /* 用户消息右对齐：靠右 + 头像在右 + 文字右对齐 */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
    }
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p {
        text-align: right !important;
    }
    /* 联网搜索结果长 URL 自动换行 */
    .stAlertContainer [data-testid="stMarkdownContainer"] {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* 使用说明 expander 右侧打开网页 SVG 图标 */
    section[data-testid="stSidebar"] details summary > span {
        display: flex !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] details summary > span > div:last-child {
        display: flex !important;
        align-items: center !important;
        flex: 1 !important;
    }
    section[data-testid="stSidebar"] details summary > span > div:last-child::after {
        content: "" !important;
        display: inline-block !important;
        width: 18px !important;
        height: 18px !important;
        margin-left: auto !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6'%3E%3C/path%3E%3Cpolyline points='15 3 21 3 21 9'%3E%3C/polyline%3E%3Cline x1='10' y1='14' x2='21' y2='3'%3E%3C/line%3E%3C/svg%3E") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        flex-shrink: 0 !important;
        cursor: pointer !important;
    }
</style>""", unsafe_allow_html=True)

# --------------------------
# 静态文件 HTTP 服务器（为 about.html 提供 HTTP 访问）
# --------------------------
_STATIC_PORT = 8765
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

class _StaticHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_STATIC_DIR, **kwargs)

def _start_static_server():
    try:
        server = socketserver.TCPServer(("127.0.0.1", _STATIC_PORT), _StaticHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError:
        pass  # 端口已被占用，服务器已在运行

_start_static_server()

# --------------------------
# 持久化 JS：通过 iframe 注入，不受 Streamlit React 重渲染影响
# --------------------------
components.html(r"""
<script>
    const doc = window.parent.document;

    function isDarkMode() {
        // 优先检查 Streamlit 的官方主题设置
        const appEl = doc.querySelector('.stApp');
        if (appEl) {
            const theme = appEl.getAttribute('data-theme');
            if (theme === 'dark') return true;
            if (theme === 'light') return false; // 明确的浅色主题设置
        }
        // 检查文档根元素和 body
        const rootTheme = doc.documentElement.getAttribute('data-theme');
        const bodyTheme = doc.body.getAttribute('data-theme');
        if (rootTheme === 'dark' || bodyTheme === 'dark') return true;
        if (rootTheme === 'light' || bodyTheme === 'light') return false;

        // 最后才检查系统偏好和背景色（作为后备）
        // 但优先级低于 Streamlit 的明确主题设置
        const bodyBg = window.parent.getComputedStyle(doc.body).backgroundColor;
        if (bodyBg) {
            const m = bodyBg.match(/\d+/g);
            if (m && m.length >= 3) {
                const r = parseInt(m[0]), g = parseInt(m[1]), b = parseInt(m[2]);
                // 提高阈值到 128，更准确地判断浅色/深色背景
                // 浅色背景通常是 RGB > 200，深色背景 < 128
                if (r + g + b < 128) return true;
            }
        }
        return false;
    }

    function applyFixedStyles() {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        let sidebarWidth = 0;
        if (sidebar) {
            const rect = sidebar.getBoundingClientRect();
            sidebarWidth = (rect.width > 50) ? rect.width : 0;
        }

        const leftVal = sidebarWidth + 'px';
        const widthVal = 'calc(100vw - ' + sidebarWidth + 'px)';
        const dark = isDarkMode();
        const bgVal = dark
            ? 'linear-gradient(180deg, rgba(14,17,23,0.97) 0%, rgba(14,17,23,1) 30%)'
            : 'linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,1) 30%)';

        const targets = doc.querySelectorAll('.st-key-fixed_input');
        targets.forEach(el => {
            el.style.setProperty('position', 'fixed', 'important');
            el.style.setProperty('bottom', '0', 'important');
            el.style.setProperty('left', leftVal, 'important');
            el.style.setProperty('width', widthVal, 'important');
            el.style.setProperty('max-width', widthVal, 'important');
            el.style.setProperty('background', bgVal, 'important');
            el.style.setProperty('backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('-webkit-backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('padding', '14px 24px 18px 24px', 'important');
            el.style.setProperty('z-index', '999', 'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
            el.style.setProperty('transition', 'left 0.3s ease, width 0.3s ease, max-width 0.3s ease, background 0.3s ease', 'important');
        });

        const chatInputs = doc.querySelectorAll('.st-key-fixed_input .stChatInput > div');
        const inputBg = dark ? 'rgba(30,37,48,0.9)' : 'transparent';
        const inputBorder = dark ? '1px solid #3D4F6F' : '1px solid #E0E0E0';
        chatInputs.forEach(el => {
            el.style.setProperty('background', inputBg, 'important');
            el.style.setProperty('border', inputBorder, 'important');
            el.style.setProperty('border-radius', '24px', 'important');
            el.style.setProperty('box-shadow', 'none', 'important');
        });

        const chatTextareas = doc.querySelectorAll('.st-key-fixed_input textarea[data-testid="stChatInputTextArea"]');
        const taBg = dark ? 'rgb(38, 39, 48)' : 'rgb(255, 255, 255)';
        const taColor = dark ? 'rgb(250, 250, 250)' : 'rgb(49, 51, 63)';
        const taCaret = dark ? 'rgb(250, 250, 250)' : 'rgb(49, 51, 63)';
        chatTextareas.forEach(el => {
            el.style.setProperty('background-color', taBg, 'important');
            el.style.setProperty('color', taColor, 'important');
            el.style.setProperty('caret-color', taCaret, 'important');
        });

        const tablists = doc.querySelectorAll('div[role="radiogroup"][aria-label="导航"]');
        const tabBg = dark
            ? 'linear-gradient(180deg, rgba(14,17,23,0.97) 0%, rgba(14,17,23,1) 30%)'
            : 'linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,1) 30%)';
        const tabBorder = dark ? '1px solid #2D3748' : '1px solid #E0E0E0';
        const tabShadow = dark ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.04)';
        tablists.forEach(el => {
            el.style.setProperty('position', 'fixed', 'important');
            el.style.setProperty('top', '3.5rem', 'important');
            el.style.setProperty('left', leftVal, 'important');
            el.style.setProperty('right', '0', 'important');
            el.style.setProperty('width', widthVal, 'important');
            el.style.setProperty('background', tabBg, 'important');
            el.style.setProperty('border-bottom', tabBorder, 'important');
            el.style.setProperty('box-shadow', tabShadow, 'important');
            el.style.setProperty('backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('-webkit-backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('padding', '0 24px', 'important');
            el.style.setProperty('margin', '0', 'important');
            el.style.setProperty('z-index', '998', 'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
            el.style.setProperty('transition', 'left 0.3s ease, right 0.3s ease, width 0.3s ease, background 0.3s ease', 'important');
        });

        const tabBorders = doc.querySelectorAll('div[data-baseweb="tab-border"]');
        tabBorders.forEach(el => {
            el.style.setProperty('display', 'none', 'important');
        });

        const mainHeaders = doc.querySelectorAll('.main-header');
        mainHeaders.forEach(header => {
            const toolbar = doc.querySelector('.stAppToolbar');
            if (toolbar && header.parentNode !== toolbar) {
                const rightSection = toolbar.querySelector('.st-emotion-cache-scp8yw') || toolbar.lastElementChild;
                if (rightSection && rightSection.parentNode === toolbar) {
                    toolbar.insertBefore(header, rightSection);
                } else {
                    toolbar.appendChild(header);
                }
            }
        });

        const headerBg = dark ? 'rgba(14, 17, 23, 0.85)' : 'rgba(255, 255, 255, 0.85)';
        const headerColor = dark ? '#E3F2FD' : '#1E3A5F';
        mainHeaders.forEach(el => {
            el.style.setProperty('position', 'absolute', 'important');
            el.style.setProperty('top', '0', 'important');
            el.style.setProperty('left', '2.75rem', 'important');
            el.style.setProperty('right', '13rem', 'important');
            el.style.setProperty('height', '3.5rem', 'important');
            el.style.setProperty('line-height', '3.5rem', 'important');
            el.style.setProperty('font-size', '1.15rem', 'important');
            el.style.setProperty('font-weight', '700', 'important');
            el.style.setProperty('color', headerColor, 'important');
            el.style.setProperty('background', headerBg, 'important');
            el.style.setProperty('border-bottom', 'none', 'important');
            el.style.setProperty('box-shadow', 'none', 'important');
            el.style.setProperty('backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('-webkit-backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('padding', '0 16px', 'important');
            el.style.setProperty('margin', '0', 'important');
            el.style.setProperty('z-index', '1', 'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
            el.style.setProperty('pointer-events', 'none', 'important');
            el.style.setProperty('text-align', 'center', 'important');
            el.style.setProperty('white-space', 'nowrap', 'important');
            el.style.setProperty('overflow', 'hidden', 'important');
            el.style.setProperty('text-overflow', 'ellipsis', 'important');
        });

        // 使用说明 expander 右侧图标点击打开 about.html
        const sidebarDetails = doc.querySelectorAll('section[data-testid="stSidebar"] details');
        sidebarDetails.forEach(details => {
            const summary = details.querySelector('summary');
            if (summary && summary.textContent.includes('使用说明')) {
                const lastDiv = summary.querySelector(':scope > span > div:last-child');
                if (lastDiv && !lastDiv.dataset.aboutClickBound) {
                    lastDiv.dataset.aboutClickBound = '1';
                    lastDiv.addEventListener('click', (e) => {
                        // 仅当点击落在 18×18 图标区域内才触发
                        const rect = lastDiv.getBoundingClientRect();
                        const iconW = 18, iconH = 18;
                        const iconLeft = rect.right - iconW;
                        const iconTop = rect.top + (rect.height - iconH) / 2;
                        if (e.clientX >= iconLeft && e.clientX <= rect.right &&
                            e.clientY >= iconTop && e.clientY <= iconTop + iconH) {
                            e.stopPropagation();
                            e.preventDefault();
                            window.open('http://127.0.0.1:8765/about.html', '_blank');
                        }
                    });
                }
            }
        });
    }

    setInterval(applyFixedStyles, 300);
    applyFixedStyles();
    setTimeout(applyFixedStyles, 100);
    setTimeout(applyFixedStyles, 500);
    setTimeout(applyFixedStyles, 1000);

    const tryObserve = () => {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            const ro = new ResizeObserver(applyFixedStyles);
            ro.observe(sidebar);
            const mo = new MutationObserver(applyFixedStyles);
            mo.observe(sidebar, { attributes: true, attributeFilter: ['aria-expanded', 'class', 'style'] });
        } else {
            setTimeout(tryObserve, 300);
        }
        const appEl = doc.querySelector('.stApp');
        if (appEl) {
            const themeMo = new MutationObserver(applyFixedStyles);
            themeMo.observe(appEl, { attributes: true, attributeFilter: ['data-theme', 'class'] });
        }
    };
    setTimeout(tryObserve, 200);

    window.parent.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyFixedStyles);
    window.parent.addEventListener('resize', applyFixedStyles);
</script>
""", height=0)

# --------------------------
# 重启确认对话框（真实进程重启）
# --------------------------
@st.dialog("系统重启")
def restart_dialog(message: str):
    st.markdown("""
    <style>
        [data-testid="stDialog"] button:has(svg) {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.warning(message)
    if st.button("确认重启", use_container_width=True):
        # 真实进程重启：启动全新的独立 Streamlit 进程，然后退出当前进程
        port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        address = os.environ.get("STREAMLIT_SERVER_ADDRESS", "localhost")
        script_path = os.path.abspath(__file__)

        cmd = [
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", port,
            "--server.address", address,
        ]

        if sys.platform == "win32":
            # DETACHED_PROCESS: 新进程完全独立，不继承父进程控制台
            env = os.environ.copy()
            env["SKIP_LOADER"] = "1"
            subprocess.Popen(cmd, creationflags=0x00000008 | 0x08000000, env=env)
        else:
            env = os.environ.copy()
            env["SKIP_LOADER"] = "1"
            subprocess.Popen(cmd, start_new_session=True, env=env)

        # 立即终止当前进程
        os._exit(0)

@st.dialog("确认删除")
def delete_file_dialog(filename: str):
    st.error(f"确认删除 **{filename}** ？")
    _, btn_col = st.columns([2, 1])
    with btn_col:
        if st.button("确认删除", key=f"confirm_del_{filename}", use_container_width=True):
            try:
                file_path = os.path.join("knowledge_base", filename)
                os.remove(file_path)
                new_count = st.session_state.rag.load_knowledge_base()
                st.session_state.knowledge_count = new_count
                st.session_state.show_restart_dialog = True
                st.session_state.restart_message = "文件已删除，需要重启系统以完成初始化，是否立即重启？"
                st.rerun()
            except Exception as e:
                st.error(f"❌ 删除 {filename} 失败: {e}")

@st.dialog("确认批量删除")
def batch_delete_files_dialog(files: list):
    st.error(f"确认删除以下 {len(files)} 个文件？")
    for f in files:
        st.write(f"• {f}")
    _, btn_col = st.columns([2, 1])
    with btn_col:
        if st.button("确认删除", key="confirm_batch_del", use_container_width=True):
            try:
                for f in files:
                    file_path = os.path.join("knowledge_base", f)
                    os.remove(file_path)
                new_count = st.session_state.rag.load_knowledge_base()
                st.session_state.knowledge_count = new_count
                st.session_state.show_restart_dialog = True
                st.session_state.restart_message = f"已删除 {len(files)} 个文件，需要重启系统以完成初始化，是否立即重启？"
                st.rerun()
            except Exception as e:
                st.error(f"❌ 批量删除失败: {e}")

# --------------------------
# 初始化检查
# --------------------------
create_directories()

# --------------------------
# 配置验证
# --------------------------
errors = validate_config()
if errors:
    st.error("❌ 配置错误")
    for e in errors:
        st.warning(f"• {e}")
    st.info("请检查 `.env` 文件中的配置项")
    st.stop()

# --------------------------
# 单例初始化（只加载1次）
# --------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.init_error = None
    # 对话历史存储
    st.session_state.rag_chat_history = []  # 知识问答对话历史
    st.session_state.agent_chat_history = []  # 智能体对话历史
    st.session_state.rag_session_id = "rag_default"
    st.session_state.agent_session_id = "agent_default"
    st.session_state.max_history = 10  # 最大记忆轮数
    st.session_state.rag_search_mode = "hybrid"  # 检索模式：keyword / semantic / hybrid
    

if not st.session_state.initialized:
    skip_loader = os.environ.get("SKIP_LOADER") == "1"
    os.environ.pop("SKIP_LOADER", None)  # 只跳过一次

    hide_header = None
    loader_placeholder = None

    if not skip_loader:
        # 隐藏 Streamlit 顶栏
        hide_header = st.empty()
        hide_header.markdown("""
            <style>
                [data-testid="stHeader"] { display: none !important; }
                [data-testid="stToolbar"] { display: none !important; }
                #MainMenu { display: none !important; }
                footer { display: none !important; }
            </style>
        """, unsafe_allow_html=True)

        # 自定义书本翻页加载动画
        loader_placeholder = st.empty()
        loader_html_path = os.path.join(os.path.dirname(__file__), "static", "book-loader.html")
        with open(loader_html_path, "r", encoding="utf-8") as f:
            loader_html = f.read()
        with loader_placeholder:
            st.html(loader_html)

    try:
        st.session_state.rag = RAGModule(max_history=st.session_state.max_history)
        st.session_state.agent = AgentModule(rag_module=st.session_state.rag, max_history=st.session_state.max_history)

        # 加载知识库
        count = st.session_state.rag.load_knowledge_base()
        st.session_state.knowledge_count = count

        st.session_state.initialized = True
        st.session_state.init_error = None
        if not skip_loader:
            time.sleep(10)
            loader_placeholder.empty()
            hide_header.empty()
    except Exception as e:
        if loader_placeholder:
            loader_placeholder.empty()
        if hide_header:
            hide_header.empty()
        st.session_state.init_error = str(e)
        st.error(f"❌ 系统初始化失败: {e}")
        st.stop()

# --------------------------
# 侧边栏
# --------------------------
with st.sidebar:
    # 系统状态
    st.subheader("⚙️ 系统状态")
    if st.session_state.initialized:
        st.success("✅ 已初始化")
        st.info(f"📚 知识库片段: {st.session_state.knowledge_count}")
    else:
        st.warning("⚠️ 初始化中...")

    st.markdown("---")
    
    # 功能说明
    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        **📖\ufe0f 知识问答**
        - 基于劳动与社会保障法知识库进行智能回答
        - 涵盖劳动合同、社会保险、劳动争议、工资福利等
        
        **🤖\ufe0f 智能体**
        - 真正的 ReAct 智能体，可自主调用工具
        - 自动检索知识库 → 联网搜索 → 赔偿计算
        - 🧠 深度思考模式：多角度检索、交叉验证、结构化分析
        - 展示完整推理链（工具调用 + 结果）
        - 支持多轮对话记忆
        
        **📁\ufe0f 文件上传**
        - 支持 PDF、DOCX、TXT 格式
        - 自动加入知识库
        """)
    
    st.markdown("---")

    # 记忆管理
    st.subheader("💭 对话记忆管理")

    col1, col2 = st.columns(2)
    with col1:
        rag_info = st.session_state.rag.get_session_info(st.session_state.rag_session_id) if st.session_state.initialized else {"turn_count": 0}
        st.metric("知识问答", f"{rag_info['turn_count']} 轮")
    with col2:
        agent_info = st.session_state.agent.get_session_info(st.session_state.agent_session_id) if st.session_state.initialized else {"turn_count": 0}
        st.metric("智能体", f"{agent_info['turn_count']} 轮")

    if st.button("🗑️ 清空所有对话记忆", use_container_width=True):
        if st.session_state.initialized:
            st.session_state.rag.clear_history(st.session_state.rag_session_id)
            st.session_state.agent.clear_history(st.session_state.agent_session_id)
            st.session_state.rag_chat_history = []
            st.session_state.agent_chat_history = []
            st.success("✅ 已清空所有对话记忆")
            st.rerun()

    st.markdown("---")
    st.caption("© 2026 劳动与社会保障法律法规知识智能助手")

# --------------------------
# 主界面
# --------------------------
st.markdown('<div class="main-header">⚖️ 劳动与社会保障法律法规知识智能助手</div>', unsafe_allow_html=True)
st.markdown("---")

tab_options = ["📖\ufe0f 知识问答", "🤖\ufe0f 智能体", "📁\ufe0f 文件上传"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_options[0]
active_tab = st.radio(
    "导航",
    tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab_radio",
)
st.session_state.active_tab = active_tab

# ===== 底部固定输入区（Form 始终在 DOM 内，所有 Tab 结构一致，不闪） =====
rag_question = ""
agent_task = ""
submitted = False

with st.container(key="fixed_input"):
    with st.form(key="shared_input_form", clear_on_submit=True):
        if active_tab == "📖\ufe0f 知识问答":
            ctrl_cols = st.columns([3, 3, 19], gap="small")
            with ctrl_cols[0]:
                st.checkbox("🌐 联网搜索", value=False, key="rag_web_search", help="结合联网搜索结果进行回答")
            with ctrl_cols[1]:
                st.session_state.rag_search_mode = st.selectbox(
                    "检索模式",
                    options=["hybrid", "semantic", "keyword"],
                    format_func=lambda x: {"hybrid": "🔀 混合", "semantic": "🧠 语义", "keyword": "🔑 关键词"}[x],
                    index=["hybrid", "semantic", "keyword"].index(st.session_state.rag_search_mode),
                    key="rag_search_mode_radio",
                    help="混合检索兼顾精度与召回；语义检索基于向量相似度；关键词检索无需 Embedding 模型",
                    label_visibility="collapsed",
                )

            input_cols = st.columns([11, 1])
            with input_cols[0]:
                st.text_input(
                    "请输入您的法律问题",
                    key="rag_text_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的法律问题...",
                )
            with input_cols[1]:
                submitted = st.form_submit_button("➤", use_container_width=True, help="发送（Enter）")

        elif active_tab == "🤖\ufe0f 智能体":
            ctrl_cols = st.columns([3, 3, 19], gap="small")
            with ctrl_cols[0]:
                st.checkbox("🧠 深度思考", value=False, key="agent_deep_thinking", help="启用深度分析模式：Agent 会从多个角度检索、交叉验证、精确计算，给出结构化分析报告")
            with ctrl_cols[1]:
                st.checkbox("🌐 联网搜索", value=False, key="agent_web_search", help="允许 Agent 联网搜索最新法规政策")

            input_cols = st.columns([11, 1])
            with input_cols[0]:
                st.text_input(
                    "请输入您的任务",
                    key="agent_text_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的任务...",
                )
            with input_cols[1]:
                submitted = st.form_submit_button("➤", use_container_width=True, help="发送（Enter）")

        else:
            # 文件上传：文件选择器和上传按钮在底部固定区并排
            upload_cols = st.columns([8.6, 1.4], gap="small")
            with upload_cols[0]:
                st.file_uploader(
                    "选择文件",
                    type=["pdf", "docx", "txt", "md"],
                    accept_multiple_files=True,
                    key=f"file_uploader_{st.session_state.get('uploader_key', 0)}",
                    label_visibility="collapsed",
                )
            with upload_cols[1]:
                submitted = st.form_submit_button("📥 上传并添加到知识库", use_container_width=True)

            if submitted and active_tab == "📁\ufe0f 文件上传":
                uploaded_files = st.session_state.get(f"file_uploader_{st.session_state.get('uploader_key', 0)}", [])
                if uploaded_files:
                    st.toast("📤 正在上传文件...")
                    success_count = 0
                    for file in uploaded_files:
                        try:
                            file_path = os.path.join("knowledge_base", file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            success_count += 1
                        except Exception as e:
                            st.error(f"❌ 上传 {file.name} 失败: {e}")

                    if success_count > 0:
                        st.toast("🔄 正在更新知识库...")
                        new_count = st.session_state.rag.load_knowledge_base()
                        st.session_state.knowledge_count = new_count

                        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1

                        st.session_state.show_restart_dialog = True
                        st.session_state.restart_message = f"文件已上传并更新知识库（当前共 {new_count} 个文本片段），需要重启系统以完成初始化，是否立即重启？"
                        st.rerun()

# 统一处理提交
if submitted:
    if active_tab == "📖\ufe0f 知识问答":
        rag_question = st.session_state.get("rag_text_input", "").strip()
    elif active_tab == "🤖\ufe0f 智能体":
        agent_task = st.session_state.get("agent_text_input", "").strip()

# ========== RAG 问答 ==========
if active_tab == tab_options[0]:
    st.subheader("📖 基于知识库的智能问答")

    # 对话历史滚动容器
    chat_container = st.container()

    with chat_container:
        # 显示对话历史
        if st.session_state.rag_chat_history:
            st.markdown("### 💬 对话历史")
            for msg in st.session_state.rag_chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="🙋"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="⚖️"):
                        web_results = msg.get("web_results") or []
                        web_error = msg.get("web_error")
                        if web_error:
                            st.warning(f"⚠️ 联网搜索失败：{web_error}")
                        if web_results:
                            links_text = "\n".join(
                                [f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)]
                            )
                            with st.expander(f"🌐 联网搜索结果（共 {len(web_results)} 条）", expanded=False):
                                st.info(links_text)
                        st.markdown(msg["content"])
        

        # 知识库状态提示
        if st.session_state.knowledge_count == 0:
            st.markdown('<div class="warning-box">⚠️ 知识库为空，请先上传文档！</div>', unsafe_allow_html=True)
        

    # 处理对话历史中的清空按钮
    if st.session_state.rag_chat_history:
        if st.button("🗑️ 清空知识问答对话", key="clear_rag_history"):
            st.session_state.rag.clear_history(st.session_state.rag_session_id)
            st.session_state.rag_chat_history = []
            st.rerun()

    # 处理用户输入
    if rag_question:
        question = rag_question
        # 添加用户消息到历史
        st.session_state.rag_chat_history.append({"role": "user", "content": question})

        # 显示用户消息
        with chat_container:
            with st.chat_message("user", avatar="🙋"):
                st.markdown(question)

            # 获取回答
            with st.chat_message("assistant", avatar="⚖️"):
                if st.session_state.rag_web_search:
                    with st.spinner("🔍 正在检索知识库和联网搜索..."):
                        try:
                            stream_gen, web_ctx = st.session_state.rag.ask_with_web_search_stream(
                                question,
                                session_id=st.session_state.rag_session_id,
                                search_mode=st.session_state.rag_search_mode,
                            )
                            web_results = web_ctx.get("web_results", [])
                            web_error = web_ctx.get("web_error")

                            if web_error:
                                st.warning(f"⚠️ 联网搜索失败：{web_error}")
                            if web_results:
                                links_text = "\n".join(
                                    [f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)]
                                )
                                with st.expander(f"🌐 联网搜索结果（共 {len(web_results)} 条）", expanded=False):
                                    st.info(links_text)

                            answer = st.write_stream(stream_gen)
                            st.session_state.rag_chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "web_results": web_results,
                                "web_error": web_error,
                            })
                        except Exception as e:
                            error_msg = f"❌ 回答失败: {e}"
                            st.error(error_msg)
                            st.session_state.rag_chat_history.append({"role": "assistant", "content": error_msg})
                else:
                    with st.spinner("🔍 正在检索知识库..."):
                        try:
                            answer = st.write_stream(st.session_state.rag.ask_stream(
                                question,
                                session_id=st.session_state.rag_session_id,
                                search_mode=st.session_state.rag_search_mode,
                            ))
                            st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            error_msg = f"❌ 回答失败: {e}"
                            st.error(error_msg)
                            st.session_state.rag_chat_history.append({"role": "assistant", "content": error_msg})

# ========== Agent 智能体 ==========
elif active_tab == tab_options[1]:
    st.subheader("🤖\ufe0f 智能体任务执行")

    # 对话历史滚动容器
    chat_container = st.container()

    with chat_container:
        # 显示对话历史
        if st.session_state.agent_chat_history:
            st.markdown("### 💬 对话历史")
            for msg in st.session_state.agent_chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="🙋"):
                        st.markdown(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        if msg.get("thinking"):
                            hist_title = "🧠 深度思考" if msg.get("deep_thinking") else "🔍 思考过程"
                            with st.expander(hist_title, expanded=False):
                                st.markdown(msg["thinking"])
                        st.markdown(msg["content"])
       

    # 处理对话历史中的清空按钮
    if st.session_state.agent_chat_history:
        if st.button("🗑️ 清空智能体对话", key="clear_agent_history"):
            st.session_state.agent.clear_history(st.session_state.agent_session_id)
            st.session_state.agent_chat_history = []
            st.rerun()

    if agent_task:
        task = agent_task
        # 添加用户消息到历史
        st.session_state.agent_chat_history.append({"role": "user", "content": task})

        # 显示用户消息
        with chat_container:
            with st.chat_message("user", avatar="🙋"):
                st.markdown(task)

            # 执行任务并显示
            with st.chat_message("assistant", avatar="🤖"):
                # 创建实时显示的容器
                thinking_placeholder = st.empty()
                answer_placeholder = st.empty()

                thinking_text = ""
                final_answer = ""

                try:
                    # 流式执行任务
                    for event in st.session_state.agent.run_task_stream(
                        task,
                        enable_web_search=st.session_state.agent_web_search,
                        deep_thinking=st.session_state.agent_deep_thinking,
                        session_id=st.session_state.agent_session_id
                    ):
                        event_type = event.get("type")
                        content = event.get("content", "")

                        # 深度思考 vs 普通模式的标题
                        think_title = "🧠 深度思考" if st.session_state.agent_deep_thinking else "🔍 推理过程（ReAct）"

                        if event_type == "thinking":
                            thinking_text = content
                            with thinking_placeholder.container():
                                with st.expander(think_title, expanded=True):
                                    st.markdown(content)

                        elif event_type == "thinking_complete":
                            with thinking_placeholder.container():
                                with st.expander(think_title, expanded=True):
                                    st.markdown(content)

                        elif event_type == "tool_start":
                            tool_name = event.get("tool", "unknown")
                            tool_input = event.get("input", "")
                            thinking_text += f"\n\n---\n\n🔧 **调用工具：{tool_name}**\n> 输入：`{tool_input[:200]}`\n"
                            with thinking_placeholder.container():
                                with st.expander(think_title, expanded=True):
                                    st.markdown(thinking_text)

                        elif event_type == "tool_end":
                            tool_name = event.get("tool", "unknown")
                            output = event.get("output", "")
                            preview = output[:500] + "..." if len(output) > 500 else output
                            thinking_text += f"\n📋 **{tool_name} 结果：**\n{preview}\n"
                            with thinking_placeholder.container():
                                with st.expander(think_title, expanded=True):
                                    st.markdown(thinking_text)

                        elif event_type == "answer_chunk":
                            final_answer = content
                            with answer_placeholder.container():
                                st.markdown("### 📋 最终答案")
                                st.markdown(content)

                        elif event_type == "answer":
                            final_answer = content
                            with answer_placeholder.container():
                                st.markdown("### 📋 最终答案")
                                st.markdown(content)

                        elif event_type == "answer_complete":
                            final_answer = content
                            # 思考过程折叠
                            with thinking_placeholder.container():
                                if thinking_text.strip():
                                    with st.expander(think_title, expanded=False):
                                        st.markdown(thinking_text)
                            # 最终答案
                            with answer_placeholder.container():
                                st.markdown("### 📋 最终答案")
                                st.markdown(content)

                        elif event_type == "error":
                            st.error(f"❌ {content}")

                    # 添加助手消息到历史
                    st.session_state.agent_chat_history.append({
                        "role": "assistant",
                        "content": final_answer,
                        "thinking": thinking_text if thinking_text.strip() else None,
                        "deep_thinking": st.session_state.agent_deep_thinking,
                    })

                except Exception as e:
                    error_msg = f"❌ 执行失败: {e}"
                    st.error(error_msg)
                    st.session_state.agent_chat_history.append({"role": "assistant", "content": error_msg})

# ========== 文件上传 ==========
elif active_tab == tab_options[2]:
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # 检查是否需要弹出重启对话框
    if st.session_state.get("show_restart_dialog"):
        restart_dialog(st.session_state.get("restart_message", ""))
    
    # 显示已有文件
    kb_path = "knowledge_base"
    if os.path.exists(kb_path):
        existing_files = [f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))]
        if existing_files:
            st.markdown("---")
            st.subheader("📁 知识库文件")

            # 全选 + 批量删除按钮
            selected = [f for f in existing_files if st.session_state.get(f"cb_{f}", False)]
            all_on = all(st.session_state.get(f"cb_{f}", False) for f in existing_files)
            col_toggle, col_batch, col_spacer = st.columns([1, 1.5, 6])
            with col_toggle:
                if st.button("取消全选" if all_on else "全选", key="toggle_all", use_container_width=True):
                    for f in existing_files:
                        st.session_state[f"cb_{f}"] = not all_on
                    st.rerun()
            with col_batch:
                if selected:
                    if st.button(f"🗑️ 批量删除 ({len(selected)})", key="batch_del_btn", type="primary", use_container_width=True):
                        batch_delete_files_dialog(selected)

            for f in existing_files:
                file_path = os.path.join(kb_path, f)
                size = os.path.getsize(file_path)
                col_cb, col1, col2 = st.columns([0.5, 4, 1])
                with col_cb:
                    st.checkbox("", key=f"cb_{f}", label_visibility="collapsed")
                with col1:
                    st.write(f"• {f} ({size} bytes)")
                with col2:
                    _, right = st.columns([1, 2])
                    with right:
                        if st.button("🗑️ 删除", key=f"del_{f}", use_container_width=True):
                            delete_file_dialog(f)
        else:
            st.info("📭 暂无文件")

# --------------------------
# 页脚
# --------------------------
st.markdown("---")
st.caption("💡 提示：请确保已正确配置 API Key，否则无法正常使用问答功能")