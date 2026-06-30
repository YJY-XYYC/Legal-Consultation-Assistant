# 知识问答 / 智能体 / 文件上传 按钮中间浮动布局设计

- 日期：2026-06-30
- 范围：仅前端布局（Streamlit UI），不影响后端 RAG / Agent / 上传逻辑
- 参考布局：DeepSeek 主页面（"欢迎页 → 对话页"过渡）

---

## 1. 设计目标

把"知识问答 / 智能体 / 文件上传"三个模式切换按钮从**顶部**（`div[role="tablist"]` 由 CSS 固定到 `top: 3.5rem`、左贴侧边栏）改为**视口正中浮动**，参考 DeepSeek 主页的"中央胶囊菜单 + 底部输入框"组合。

进一步：

- **空闲态**（还没发过消息）：三个按钮作为一组胶囊，停在视口中央
- **对话态**（任意模式下已发过至少一条用户消息）：按钮回到顶部原始位置（左贴侧边栏，与原来 `top: 3.5rem; left: 21rem` 的位置一致）
- 中间 → 顶部的过渡使用 CSS `transform / top` 平滑过渡（非突变）
- 同时支持浅色 / 深色两种 Streamlit 主题
- 底部固定输入区（`st-key-fixed_input_rag / agent`）**保持不变**

---

## 2. 状态模型

在 `st.session_state` 上新增 2 个字段：

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `has_conversation` | `bool` | `False` | 任意模式（RAG / Agent）已收到至少一条用户消息 |
| `active_mode` | `str` | `"rag"` | 当前选中的模式：`"rag"` \| `"agent"` \| `"upload"` |

派生规则：

- RAG 的 `chat_input` 处理函数里，第一条 `prompt` 提交时：若 `not st.session_state.has_conversation`，置 `True`
- Agent 的 `chat_input` 处理函数里，同样触发位置
- upload 模式不参与 `has_conversation` 切换

### 重置时机

只在以下场景重置 `has_conversation = False`：

- 用户在左侧栏点击 **"🗑️ 清空所有对话记忆"** 按钮时
- 页面重启 / 重渲染 / 深浅主题切换都**不**重置——保留"已开始对话"事实

### 客户端同步

`has_conversation` 是 Python 端的真值，但 CSS 选择器需要 HTML 上的 `data-has-conversation` 属性。同步方案：

- Python 在写入 `has_conversation = True` 时，注入一段 `components.html` 把 `window.parent.sessionStorage.setItem('RAG_HAS_CONV','1')` / `'AGENT_HAS_CONV','1'`
- 清空时同时清掉这俩 key
- JS 在每次 Streamlit rerun 后（`MutationObserver` 监 `document.body` 子树变化）读取 sessionStorage 并写到 `document.documentElement.setAttribute('data-has-conversation', 'true' | 'false')`

---

## 3. DOM 与组件

**DOM 不变**——继续使用 `st.tabs(["📖 知识问答", "🤖 智能体", "📁 文件上传"])`。每个 tab 的内容面板渲染逻辑保持不变。只有 CSS 改了 `div[role="tablist"]` 的视觉与位置。

**不新建** `st.button` / `st.radio` 作模式切换——继续用 `st.tabs` 的天然切换能力。

---

## 4. CSS（关键部分）

替换 app.py 内 `div[role="tablist"]` 的 CSS 块为：

```css
/* ---------- Tab 按钮定位 / 外观 ---------- */
div[role="tablist"] {
  position: fixed !important;
  z-index: 998 !important;
  display: flex !important;
  justify-content: center !important;
  gap: 8px !important;
  padding: 8px 12px !important;
  margin: 0 !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  background: rgba(255, 255, 255, 0.85) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  backdrop-filter: blur(14px) !important;
  border-radius: 999px !important;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08) !important;

  /* 默认：中间 */
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;

  transition: top 0.45s cubic-bezier(.4, 0, .2, 1),
              left  0.45s cubic-bezier(.4, 0, .2, 1),
              transform 0.45s cubic-bezier(.4, 0, .2, 1),
              border-radius 0.45s cubic-bezier(.4, 0, .2, 1),
              background 0.3s ease,
              box-shadow 0.3s ease !important;
}

/* 隐藏 Tab 下划线（沿用现有逻辑） */
div[data-baseweb="tab-border"] {
  display: none !important;
}

/* ---------- 深色主题适配 ---------- */
[data-theme="dark"] div[role="tablist"],
.stApp[data-theme="dark"] div[role="tablist"] {
  background: rgba(38, 39, 48, 0.85) !important;
  border-color: rgba(255, 255, 255, 0.10) !important;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4) !important;
}

/* ---------- 选中状态 ---------- */
[role="tab"][aria-selected="true"] {
  background: rgba(102, 126, 234, 0.15) !important;
}
[data-theme="dark"] [role="tab"][aria-selected="true"],
.stApp[data-theme="dark"] [role="tab"][aria-selected="true"] {
  background: rgba(118, 75, 162, 0.25) !important;
}

/* ---------- 空闲 → 对话：回到侧边栏顶部 ---------- */
:where([data-has-conversation="true"]) div[role="tablist"] {
  top: 3.5rem !important;
  left: 21rem !important;
  transform: translate(0, 0) !important;
  border-radius: 12px !important;
  background: rgba(255, 255, 255, 0.97) !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}
[data-theme="dark"]:where([data-has-conversation="true"]) div[role="tablist"],
.stApp[data-theme="dark"]:where([data-has-conversation="true"]) div[role="tablist"] {
  background: rgba(38, 39, 48, 0.97) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
}

/* ---------- 主内容区补 padding，避免被悬浮按钮遮挡 ---------- */
.main .block-container {
  /* 顶部 110px（中央/顶部 tab 都覆盖到，距 70px 足够不被压住） */
  padding-top: 110px !important;
  padding-bottom: 240px !important;
  max-height: calc(100vh - 240px) !important;
  overflow-y: auto !important;
}
```

---

## 5. JS 注入

复用现有 `components.html` script 块，在尾部追加：

```js
const ROOT_HAS_CONV_KEY = '__ROOT_HAS_CONV__';

function syncHasConv() {
  const r = window.parent.sessionStorage.getItem('RAG_HAS_CONV') === '1';
  const a = window.parent.sessionStorage.getItem('AGENT_HAS_CONV') === '1';
  const val = (r || a) ? 'true' : 'false';
  if (doc.documentElement.getAttribute('data-has-conversation') !== val) {
    doc.documentElement.setAttribute('data-has-conversation', val);
  }
}

window.parent.addEventListener('storage', syncHasConv);

// 监听 Streamlit 重渲染后 body 子树变化，重新对一次
new MutationObserver(syncHasConv).observe(doc.body, { childList: true, subtree: false });
```

---

## 6. Python 修改清单

| 文件 | 行号 / 位置 | 修改内容 |
|---|---|---|
| `app.py` | 第 439-440 行附近 | `st.session_state.has_conversation = False` 与 `st.session_state.active_mode = "rag"` 初始化 |
| `app.py` | RAG `chat_input` 处理入口 | 若 `not has_conversation`，置 `True` 并 `components.html("<script>window.parent.sessionStorage.setItem('RAG_HAS_CONV','1');</script>", height=0)` |
| `app.py` | Agent `chat_input` 处理入口 | 同上，但 key 为 `'AGENT_HAS_CONV'` |
| `app.py` | 侧边栏"🗑️ 清空所有对话记忆"按钮回调 | 多重置 `has_conversation=False` 与 `components.html("<script>window.parent.sessionStorage.removeItem('RAG_HAS_CONV');window.parent.sessionStorage.removeItem('AGENT_HAS_CONV');</script>", height=0)` |
| `app.py` | CSS 块 | 见第 4 节整段替换（仅 `div[role="tablist"]` 块；其他样式不动） |
| `app.py` | JS 块 | 见第 5 节 append 到现有 `<script>` |

---

## 7. 兼容性 / 风险

1. **z-index 冲突**：中央胶囊 `z-index: 998`，底部固定输入框 `z-index: 999`。层级正确。
2. **Modal / Toast 弹层**：Streamlit 自带的 toast / 弹层使用更高的 z-index，不受本变更影响。
3. **主题切换**：CSS 同时覆盖浅色 / 深色两套选择器。`st.data-theme="dark"` 属性是 Streamlit 注入在 `.stApp` / `<html>` 上的。
4. **侧边栏折叠**：未来用户折叠侧边栏时，原顶部布局的 `left: 21rem` 会偏移。本次变更不影响此点（与现状行为一致）。
5. **`st.tabs` 副产物**：`st.tabs` 的内容面板仍在主内容流，按原逻辑渲染，并被 `padding-top: 110px` 推开。

---

## 8. 手动验证清单

1. 应用启动后，三个按钮**居中漂浮**在视口正中
2. 进入"知识问答"，发送**第一条**消息 → 按钮**平滑上滑**到顶部侧贴
3. 切换到"智能体"，发一条以上 → 按钮仍在顶部（不重置回中央）
4. 切换到"文件上传" → 按钮仍在顶部，且不影响 tab 内容
5. 侧边栏点击"🗑️ 清空所有对话记忆" → 按钮应在 rerun 后**重置回中央**
6. 深色主题下切换同样效果，中央胶囊颜色与背景对比足够
7. 底部固定输入区（两个模式）依然贴在底部，不被新位置干扰

---

## 9. 不在本设计范围内（YAGNI）

- 不引入新组件库
- 不重构 `st.session_state` 现有字段
- 不修改后端 RAG / Agent / 上传逻辑
- 不引入 `st.fragment` 或 `st.rerun` 策略变化
