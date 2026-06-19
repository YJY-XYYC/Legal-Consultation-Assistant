import streamlit as st
import streamlit.components.v1 as components
import os
import sys

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
        border-bottom: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
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
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    /* ===== 底部固定输入区样式（JS 内联样式为主，CSS 作为 fallback） ===== */
    .st-key-fixed_input_rag,
    .st-key-fixed_input_agent {
        position: fixed !important;
        bottom: 0 !important;
        left: 21rem !important;
        width: calc(100vw - 21rem) !important;
        max-width: calc(100vw - 21rem) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,1) 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 14px 24px 18px 24px !important;
        border-top: 1px solid #E0E0E0 !important;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.08) !important;
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
    .st-key-fixed_input_rag > div,
    .st-key-fixed_input_agent > div {
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
    .st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"],
    .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"] {
        background-color: rgb(255, 255, 255) !important;
        color: rgb(49, 51, 63) !important;
    }
    /* 固定输入区内的 textarea - 暗黑模式深色背景 */
    [data-theme="dark"] .st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"],
    [data-theme="dark"] .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"],
    .stApp[data-theme="dark"] .st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"],
    .stApp[data-theme="dark"] .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"] {
        background-color: rgb(38, 39, 48) !important;
        color: rgb(250, 250, 250) !important;
        caret-color: rgb(250, 250, 250) !important;
    }
    /* placeholder 颜色 - 暗黑模式 */
    [data-theme="dark"] .st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"]::placeholder,
    [data-theme="dark"] .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"]::placeholder,
    .stApp[data-theme="dark"] .st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"]::placeholder,
    .stApp[data-theme="dark"] .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: rgba(250, 250, 250, 0.5) !important;
    }
    /* 复选框在固定区内紧凑显示 */
    .st-key-fixed_input_rag .stCheckbox,
    .st-key-fixed_input_agent .stCheckbox {
        margin-bottom: 0 !important;
    }
    /* ===== 顶部固定 Tab 栏（3 个 tab 按钮固定到页面顶部） ===== */
    div[role="tablist"] {
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
    /* 隐藏 tab 下方的红色边框指示线 */
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
    /* 隐藏所有 markdown 分割线（hr） */
    div[data-testid="stMarkdownContainer"] hr {
        display: none !important;
    }
    /* 顶部固定后，给主内容区增加上内边距，避免内容被 tab 栏遮挡 */
    .main .block-container {
        padding-top: 110px !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 持久化 JS：通过 iframe 注入，不受 Streamlit React 重渲染影响
# --------------------------
components.html(r"""
<script>
    const doc = window.parent.document;

    function isDarkMode() {
        const appEl = doc.querySelector('.stApp');
        if (appEl) {
            const theme = appEl.getAttribute('data-theme');
            if (theme === 'dark') return true;
        }
        if (doc.documentElement.getAttribute('data-theme') === 'dark') return true;
        if (doc.body.getAttribute('data-theme') === 'dark') return true;
        if (window.parent.matchMedia('(prefers-color-scheme: dark)').matches) return true;
        const bodyBg = window.parent.getComputedStyle(doc.body).backgroundColor;
        if (bodyBg) {
            const m = bodyBg.match(/\d+/g);
            if (m && m.length >= 3) {
                const r = parseInt(m[0]), g = parseInt(m[1]), b = parseInt(m[2]);
                if (r + g + b < 200) return true;
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
        const borderVal = dark ? '1px solid #2D3748' : '1px solid #E0E0E0';
        const shadowVal = dark ? '0 -4px 20px rgba(0,0,0,0.3)' : '0 -4px 20px rgba(0,0,0,0.08)';

        const targets = doc.querySelectorAll('.st-key-fixed_input_rag, .st-key-fixed_input_agent');
        targets.forEach(el => {
            el.style.setProperty('position', 'fixed', 'important');
            el.style.setProperty('bottom', '0', 'important');
            el.style.setProperty('left', leftVal, 'important');
            el.style.setProperty('width', widthVal, 'important');
            el.style.setProperty('max-width', widthVal, 'important');
            el.style.setProperty('background', bgVal, 'important');
            el.style.setProperty('border-top', borderVal, 'important');
            el.style.setProperty('box-shadow', shadowVal, 'important');
            el.style.setProperty('backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('-webkit-backdrop-filter', 'blur(12px)', 'important');
            el.style.setProperty('padding', '14px 24px 18px 24px', 'important');
            el.style.setProperty('z-index', '999', 'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
            el.style.setProperty('transition', 'left 0.3s ease, width 0.3s ease, max-width 0.3s ease, background 0.3s ease', 'important');
        });

        const chatInputs = doc.querySelectorAll('.st-key-fixed_input_rag .stChatInput > div, .st-key-fixed_input_agent .stChatInput > div');
        const inputBg = dark ? 'rgba(30,37,48,0.9)' : 'transparent';
        const inputBorder = dark ? '1px solid #3D4F6F' : '1px solid #E0E0E0';
        chatInputs.forEach(el => {
            el.style.setProperty('background', inputBg, 'important');
            el.style.setProperty('border', inputBorder, 'important');
            el.style.setProperty('border-radius', '24px', 'important');
            el.style.setProperty('box-shadow', 'none', 'important');
        });

        const chatTextareas = doc.querySelectorAll('.st-key-fixed_input_rag textarea[data-testid="stChatInputTextArea"], .st-key-fixed_input_agent textarea[data-testid="stChatInputTextArea"]');
        const taBg = dark ? 'rgb(38, 39, 48)' : 'rgb(255, 255, 255)';
        const taColor = dark ? 'rgb(250, 250, 250)' : 'rgb(49, 51, 63)';
        const taCaret = dark ? 'rgb(250, 250, 250)' : 'rgb(49, 51, 63)';
        chatTextareas.forEach(el => {
            el.style.setProperty('background-color', taBg, 'important');
            el.style.setProperty('color', taColor, 'important');
            el.style.setProperty('caret-color', taCaret, 'important');
        });

        const tablists = doc.querySelectorAll('div[role="tablist"]');
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
        const headerBorder = dark ? '1px solid #2D3748' : '1px solid #E0E0E0';
        const headerShadow = dark ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.04)';
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
            el.style.setProperty('border-bottom', headerBorder, 'important');
            el.style.setProperty('box-shadow', headerShadow, 'important');
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

if not st.session_state.initialized:
    with st.spinner("🔄 正在初始化系统..."):
        try:
            st.session_state.rag = RAGModule(max_history=st.session_state.max_history)
            st.session_state.agent = AgentModule(max_history=st.session_state.max_history)

            # 加载知识库
            count = st.session_state.rag.load_knowledge_base()
            st.session_state.knowledge_count = count

            st.session_state.initialized = True
            st.session_state.init_error = None
        except Exception as e:
            st.session_state.init_error = str(e)
            st.error(f"❌ 系统初始化失败: {e}")
            st.stop()

# --------------------------
# 侧边栏
# --------------------------
with st.sidebar:
    st.title("⚖️ 劳动与社会保障法律法规知识智能助手")
    st.markdown("---")
    
    # 系统状态
    st.subheader("系统状态")
    if st.session_state.initialized:
        st.success("✅ 已初始化")
        st.info(f"📚 知识库片段: {st.session_state.knowledge_count}")
    else:
        st.warning("⚠️ 初始化中...")

    st.markdown("---")
    
    # 功能说明
    st.subheader("使用说明")
    st.markdown("""
    **📖 知识问答**
    - 基于劳动与社会保障法知识库进行智能回答
    - 涵盖劳动合同、社会保险、劳动争议、工资福利等
    
    **🤖 智能体**
    - 分析复杂劳动与社会保障法律问题
    - 支持深度思考模式
    - 可查看推理步骤
    - 支持联网搜索最新法规政策
    
    **📁 文件上传**
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

tab1, tab2, tab3 = st.tabs(["📖 知识问答", "🤖 智能体", "📁 文件上传"])

# ========== RAG 问答 ==========
with tab1:
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

    # ===== 底部固定输入区（通过 CSS 定位 st.container 自身） =====
    fixed_container = st.container(key="fixed_input_rag")

    with fixed_container:
        ctrl_cols = st.columns([1, 3])
        with ctrl_cols[0]:
            enable_web = st.checkbox("🌐 联网搜索", value=False, key="rag_web_search", help="结合联网搜索结果进行回答")

        with st.form(key="rag_input_form", clear_on_submit=True):
            input_cols = st.columns([11, 1])
            with input_cols[0]:
                question = st.text_input(
                    "请输入您的法律问题",
                    key="rag_text_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的法律问题...",
                )
            with input_cols[1]:
                submitted = st.form_submit_button("➤", use_container_width=True, help="发送（Enter）")
        # 取出提交时的内容（form 提交后会自动清空，所以用 session_state 兜底）
        if submitted:
            question = st.session_state.get("rag_text_input", "").strip()
        else:
            question = ""

    # 处理用户输入
    if question:
        # 添加用户消息到历史
        st.session_state.rag_chat_history.append({"role": "user", "content": question})

        # 显示用户消息
        with chat_container:
            with st.chat_message("user", avatar="🙋"):
                st.markdown(question)

            # 获取回答
            with st.chat_message("assistant", avatar="⚖️"):
                if enable_web:
                    with st.spinner("🔍 正在检索知识库和联网搜索..."):
                        try:
                            result = st.session_state.rag.ask_with_web_search(
                                question,
                                session_id=st.session_state.rag_session_id
                            )
                            answer = result["answer"] if isinstance(result, dict) else result
                            web_results = result.get("web_results", []) if isinstance(result, dict) else []
                            web_error = result.get("web_error") if isinstance(result, dict) else None

                            if web_error:
                                st.warning(f"⚠️ 联网搜索失败：{web_error}")
                            if web_results:
                                links_text = "\n".join(
                                    [f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)]
                                )
                                with st.expander(f"🌐 联网搜索结果（共 {len(web_results)} 条）", expanded=False):
                                    st.info(links_text)

                            st.markdown(answer)
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
                            answer = st.session_state.rag.ask(
                                question,
                                session_id=st.session_state.rag_session_id
                            )
                            st.markdown(answer)
                            st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            error_msg = f"❌ 回答失败: {e}"
                            st.error(error_msg)
                            st.session_state.rag_chat_history.append({"role": "assistant", "content": error_msg})

# ========== Agent 智能体 ==========
with tab2:
    st.subheader("🤖 智能体任务执行")

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
                            with st.expander("🔍 思考过程", expanded=False):
                                st.markdown(msg["thinking"])
                        st.markdown(msg["content"])
       

    # 处理对话历史中的清空按钮
    if st.session_state.agent_chat_history:
        if st.button("🗑️ 清空智能体对话", key="clear_agent_history"):
            st.session_state.agent.clear_history(st.session_state.agent_session_id)
            st.session_state.agent_chat_history = []
            st.rerun()

    # ===== 底部固定输入区（通过 CSS 定位 st.container 自身） =====
    fixed_container = st.container(key="fixed_input_agent")

    with fixed_container:
        deep_thinking = st.checkbox("🧠 深度思考", value=False, key="agent_deep_thinking", help="显示详细的推理过程")
        enable_web = st.checkbox("🌐 联网搜索", value=False, key="agent_web_search", help="结合联网搜索结果")

        with st.form(key="agent_input_form", clear_on_submit=True):
            input_cols = st.columns([11, 1])
            with input_cols[0]:
                _agent_draft = st.text_input(
                    "请输入您的任务",
                    key="agent_text_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的任务...",
                )
            with input_cols[1]:
                _agent_submitted = st.form_submit_button("➤", use_container_width=True, help="发送（Enter）")
        if _agent_submitted:
            task = st.session_state.get("agent_text_input", "").strip()
        else:
            task = ""

    if task:
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
                web_placeholder = st.empty()
                answer_placeholder = st.empty()

                thinking_process = ""
                final_answer = ""

                try:
                    # 流式执行任务
                    for event in st.session_state.agent.run_task_stream(
                        task,
                        deep_thinking=deep_thinking,
                        enable_web_search=enable_web,
                        session_id=st.session_state.agent_session_id
                    ):
                        event_type = event.get("type")
                        content = event.get("content", "")

                        if event_type == "thinking":
                            thinking_process = content
                            with thinking_placeholder.container():
                                with st.expander("🔍 思考过程（实时生成中...）", expanded=True):
                                    st.markdown('<div class="thinking-box">', unsafe_allow_html=True)
                                    st.markdown(content)
                                    st.markdown('</div>', unsafe_allow_html=True)

                        elif event_type == "thinking_complete":
                            thinking_process = content
                            with thinking_placeholder.container():
                                with st.expander("🔍 思考过程", expanded=False):
                                    st.markdown('<div class="thinking-box">', unsafe_allow_html=True)
                                    st.markdown(content)
                                    st.markdown('</div>', unsafe_allow_html=True)

                        elif event_type == "web_search":
                            with web_placeholder.container():
                                with st.expander("🌐 联网搜索结果", expanded=False):
                                    st.info(content)

                        elif event_type == "web_search_error":
                            with web_placeholder.container():
                                st.warning(f"⚠️ 联网搜索失败：{content}")

                        elif event_type == "answer":
                            final_answer = content
                            with answer_placeholder.container():
                                st.markdown("### 📋 最终答案（实时生成中...）")
                                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                                st.markdown(content)
                                st.markdown('</div>', unsafe_allow_html=True)

                        elif event_type == "answer_complete":
                            final_answer = content
                            with answer_placeholder.container():
                                st.markdown("### 📋 最终答案")
                                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                                st.markdown(content)
                                st.markdown('</div>', unsafe_allow_html=True)

                        elif event_type == "error":
                            st.error(f"❌ {content}")

                    # 添加助手消息到历史
                    st.session_state.agent_chat_history.append({
                        "role": "assistant",
                        "content": final_answer,
                        "thinking": thinking_process if deep_thinking else None
                    })

                except Exception as e:
                    error_msg = f"❌ 执行失败: {e}"
                    st.error(error_msg)
                    st.session_state.agent_chat_history.append({"role": "assistant", "content": error_msg})

# ========== 文件上传 ==========
with tab3:
    st.subheader("📁 上传文档到知识库")
    
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📤 已选择 {len(uploaded_files)} 个文件")
        
        # 显示文件列表
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
        
        if st.button("📥 上传并添加到知识库"):
            with st.spinner("📤 正在上传文件..."):
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
                    # 重新加载知识库
                    with st.spinner("🔄 正在更新知识库..."):
                        new_count = st.session_state.rag.load_knowledge_base()
                        st.session_state.knowledge_count = new_count
                    
                    st.success(f"✅ 成功上传 {success_count} 个文件，知识库已更新")
                    st.info(f"📚 更新后知识库包含 {new_count} 个文本片段")
    
    # 显示已有文件
    kb_path = "knowledge_base"
    if os.path.exists(kb_path):
        existing_files = [f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))]
        if existing_files:
            st.markdown("---")
            st.subheader("📂 知识库文件")
            for f in existing_files:
                file_path = os.path.join(kb_path, f)
                size = os.path.getsize(file_path)
                st.write(f"• {f} ({size} bytes)")

# --------------------------
# 页脚
# --------------------------
st.markdown("---")
st.caption("💡 提示：请确保已正确配置 API Key，否则无法正常使用问答功能")