# 劳动与社会保障法律法规知识智能助手


## 项目简介

本项目实现了一个基于LangChain框架的劳动与社会保障法律法规领域的智能问答系统，结合RAG（检索增强生成）与ReAct 风格智能体两大核心能力，底层使用DeepSeek大模型与Tavily联网搜索，提供专业、可靠、可解释的法律咨询服务。

系统提供两套交互入口：

- 🖥️ Web 界面（推荐） — 基于 Streamlit 的现代化交互界面
- 💻 命令行模式 — 终端交互式问答

---

## 📑 目录

- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [使用指南](#-使用指南)
- [配置说明](#-配置说明)
- [核心模块](#-核心模块)
- [示例问答](#-示例问答)
- [常见问题](#-常见问题)
- [注意事项](#-注意事项)
- [路线图](#-路线图)

---

## ✨ 核心特性

### 📖 RAG 知识问答

| 能力 | 说明 |
|------|------|
| 三模检索 | 关键词 / 语义 / 混合（RRF 融合）可切换 |
| 向量检索 | Chroma + HuggingFace Embedding 语义召回 |
| RRF 融合 | 倒数排名融合算法，同时利用词法 + 语义通道 |
| 文档解析 | 支持 PDF / DOCX / TXT / MD 多格式 |
| 自动分块 | `chunk_size=500`，`overlap=50`，保留上下文 |
| 来源追溯 | 引用知识库文档，输出可溯源 |
| 联网增强 | 可选启用 Tavily 搜索，融合最新法规政策 |
| 多会话记忆 | 每会话独立上下文窗口，默认 10 轮 |

### 🤖 Agent 智能体

- **ReAct 风格**推理与任务规划
- **三大核心工具**：
  - 🐍 `python_runner` — 实时运行 Python 代码
  - 🧮 `calculator` — 基于 SymPy 的精确数学计算
  - 🔍 `web_search` — 基于 DuckDuckGo 的联网检索
- **流式输出**：思考过程与最终答案分步呈现
- **可选联网**：Tavily 搜索补充最新信息

### 🧠 深度思考模式

- 可开关的思考链生成（目标 → 法规 → 方案 → 风险）
- 思考过程与最终答案**分开展示、可折叠**
- 实时流式输出，过程透明

### 📁 知识库管理

- 多文件批量上传（PDF / DOCX / TXT / MD）
- 自动解析 → 写入 `knowledge_base/` → 刷新索引
- 实时显示知识库片段数与文件列表

### 🎨 现代化 Web 界面

- 三栏布局：📖 知识问答 / 🤖 智能体 / 📁 文件上传
- 底部固定输入区，长对话不丢视野
- 浅色 / 深色模式自适应
- 侧边栏系统状态可视化、一键清空

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────┐
│                     用户交互层                          │
│   Streamlit Web 界面  /  命令行终端  (main.py)        │
└────────────────┬───────────────────────┬───────────────┘
                 │                       │
        ┌────────▼─────────┐    ┌────────▼─────────┐
        │   RAG 模块       │    │   Agent 模块     │
        │ (rag_module.py)  │    │ (agent_module.py)│
        │                  │    │                  │
        │ • 关键词检索     │    │ • ReAct 推理     │
        │ • 语义检索       │    │ • 工具调用       │
        │ • RRF 融合       │    │ • 流式输出       │
        └────────┬─────────┘    └────────┬─────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐      ┌───────▼───────┐    ┌───────▼──────┐
   │ DeepSeek │      │   Chroma DB   │    │ Tavily / DDG │
   │   LLM    │      │   向量库      │    │   联网搜索   │
   └──────────┘      └───────────────┘    └──────────────┘
```

---

## 🛠️ 技术栈

| 类别 | 选型 |
|------|------|
| LLM 框架 | LangChain 0.3.30 |
| 大语言模型 | DeepSeek Chat（`deepseek-chat`） |
| 向量数据库 | Chroma 0.4.24（`langchain-chroma` 0.1.0） |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2`（多语言 384 维） |
| 检索策略 | 关键词 / 语义 / RRF 混合 |
| 联网搜索 | Tavily Search API / DuckDuckGo |
| Web 框架 | Streamlit 1.54.0 |
| 文档解析 | pypdf、python-docx |
| 数学计算 | SymPy 1.12 |
| 运行环境 | Python 3.10+（已在 3.13.7 验证） |

---

## 🚀 快速开始

### 1. 克隆与安装

```bash
git clone <your-repo-url>
cd llm-qa-system

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在根目录编辑 `.env` 文件（可参考下方 [配置说明](#-配置说明)）：

```env
# 必填
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# 可选（默认值已可用）
DEEPSEEK_API_BASE=https://api.deepseek.com
HF_ENDPOINT=https://hf-mirror.com   # 国内推荐
```

### 3. 启动服务

**🖥️ 方式一：Web 界面（推荐）**

```bash
streamlit run app.py
```

启动后默认访问 http://localhost:8501

> 🌐 **同网段设备访问**（手机/同 WiFi 的其他电脑访问本机服务）
>
> **第 1 步**：启动时绑定到所有网卡
> ```bash
> streamlit run app.py --server.address 0.0.0.0 --server.port 8501
> ```
>
> **第 2 步**：查看本机 IP
> - Windows：`ipconfig`（查看 `IPv4 地址`）
> - Linux / macOS：`ifconfig` 或 `ip addr`
>
> **第 3 步**：同网段设备在浏览器中访问 `http://<本机IP>:8501`
>
> >
> ⚠️ **注意事项**：
> - 手机/其他设备必须与本机连接**同一 WiFi / 局域网**
> - 如被 Windows 防火墙拦截，首次访问会弹出"是否允许"提示，请选择**允许**
> - 若路由器开启了 AP 隔离，跨设备访问会被禁用，请关闭 AP 隔离

**💻 方式二：命令行模式**

```bash
python main.py
```

根据菜单提示选择 `1`（知识问答）或 `2`（智能体任务），输入 `0` 退出。

---

## 📁 项目结构

```
llm-qa-system/
├── main.py                 # 命令行入口
├── app.py                  # Streamlit Web 界面
├── config.py               # 配置加载与校验
├── requirements.txt        # 依赖清单
├── README.md               # 项目说明
├── DEMO.md                 # 演示文档
├── .env                    # 环境变量（API Key 等）
│
├── modules/
│   ├── __init__.py
│   ├── rag_module.py       # RAG 知识问答模块
│   ├── agent_module.py     # Agent 智能体模块
│   └── tools.py            # 工具定义：代码执行 / 计算器 / 联网搜索
│
├── knowledge_base/         # 法律知识库（自动加载）
│   ├── labor_law_overview.md
│   ├── labor_contract.md
│   ├── labor_dispute.md
│   └── social_security.md
│
├── chroma_db/              # Chroma 向量库持久化（首次运行后生成）
├── models_cache/           # Embedding 模型本地缓存
└── uploads/                # 上传文件临时目录（可选）
```

---

## 📖 使用指南

### 1️⃣ 知识问答模式

1. 启动时自动加载 `knowledge_base/` 下所有文档
2. 在「📖 知识问答」Tab 输入法律相关问题
3. 系统检索 Top-K 文档并结合 DeepSeek 生成答案
4. 勾选「🌐 联网搜索」可融合最新法规
5. 支持上下文连续多轮问答

### 2️⃣ 智能体模式

1. 在「🤖 智能体」Tab 输入复杂任务（如案件分析、维权方案）
2. 勾选「🧠 深度思考」可查看分步推理过程
3. 勾选「🌐 联网搜索」可补充最新信息
4. 思考过程与最终答案分开展示，可折叠查看

### 3️⃣ 文件上传

1. 进入「📁 文件上传」Tab
2. 选择 PDF / DOCX / TXT / MD 文件（支持多选）
3. 点击「📥 上传并添加到知识库」
4. 系统自动解析、分块并刷新知识库

### 4️⃣ 对话记忆管理

- 侧边栏查看两个 Tab 的对话轮数
- 点击「🗑️ 清空所有对话记忆」可一键重置
- 最大记忆轮数默认为 10，可在 `app.py` 中调整 `st.session_state.max_history`

---

## ⚙️ 配置说明

`.env` 文件完整配置项：

```env
# ============ DeepSeek API ============
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
DEEPSEEK_EMBEDDING_MODEL=deepseek-embedding

# ============ Tavily 搜索 API ============
TAVILY_API_KEY=your_tavily_api_key_here

# ============ 向量数据库（Chroma）=========
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=labor_law_kb

# ============ Embedding ============
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_CACHE_DIR=./models_cache
EMBEDDING_DEVICE=cpu
HF_ENDPOINT=https://hf-mirror.com   # 国内推荐 hf-mirror.com

# ============ RAG 检索 ============
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
SEARCH_MODE=hybrid              # semantic / keyword / hybrid
SEMANTIC_TOP_K=8
KEYWORD_TOP_K=8

# ============ 知识库路径 ============
KNOWLEDGE_BASE_PATH=./knowledge_base
UPLOADS_PATH=./uploads
```

> ⚠️ `DEEPSEEK_API_KEY` 与 `TAVILY_API_KEY` 为**必填项**。

**获取 API Key：**

| 服务 | 申请地址 |
|------|----------|
| DeepSeek | https://platform.deepseek.com/api_keys |
| Tavily | https://tavily.com/ |

---

## 🧩 核心模块

### RAG 模块（`modules/rag_module.py`）

**三种检索模式对比：**

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `keyword` | 纯关键词 + 字符匹配 + 自研评分 | 精确术语检索（如法条编号） |
| `semantic` | Chroma + Embedding 余弦相似度 | 同义词 / 口语化表达召回 |
| `hybrid` ⭐ | RRF 融合关键词 + 语义 | **默认推荐**，兼顾准确率与召回率 |

**关键实现：**

- **关键词评分**：命中关键词 +3 分，子串匹配 +1 分，共现字符 +0.5 分
- **文本分块**：`RecursiveCharacterTextSplitter`（`chunk_size=500`，`overlap=50`）
- **RRF 公式**：`score(d) = Σ 1 / (k + rank_i)`，k=60
- **多会话**：以 `session_id` 为 key 隔离上下文，使用 `deque` 限制最大轮数
- **降级策略**：Embedding 加载失败时自动切换为 `keyword` 模式

**三种检索模式效果对比：**

| 查询 | 关键词 | 语义 | 混合（RRF）|
|------|--------|------|-----------|
| 老板把我开了，n+1 怎么算 | ❌ | ✅ | ✅ 综合最优 |
| 养老保险的缴费比例 | ❌ | ✅ | ✅ 社保+补偿结合 |
| 试用期最长能签多久 | ⚠️ | ✅ | ✅ 试用期条款优先 |

### Agent 模块（`modules/agent_module.py`）

- **流式输出**：通过 `yield` 逐步返回 `thinking` / `answer` / `web_search` 事件
- **深度思考**：先独立生成思考过程，再融合思考 + 联网结果生成最终答案
- **工具调用**：在 `tools.py` 中以 LangChain `@tool` 装饰器声明

---

## 💬 示例问答

### 📖 RAG 示例

> 🙋 **用户**：养老保险的缴费比例是多少？企业需要缴纳多少？
>
> ⚖️ **助手**：根据《社会保险法》及相关规定，基本养老保险由用人单位和职工共同缴纳：
> - 用人单位：按本单位职工工资总额的 16%~20% 缴纳（具体比例由各省确定）
> - 职工个人：按本人缴费工资的 8% 缴纳
> - 灵活就业人员：按缴费基数的 20% 左右缴纳
>
> 【来源：knowledge_base/social_security.md】

### 🤖 Agent 示例（深度思考模式）

> 🙋 **用户**：分析员工被无故辞退后的维权途径和注意事项
>
> 🤖 **助手**：
>
> 🔍 **思考过程**：
> 1. 明确任务目标：员工被无故辞退，需要梳理维权途径
> 2. 分析劳动法依据：《劳动合同法》第 39/40/41 条
> 3. 评估维权途径：协商 → 调解 → 仲裁 → 诉讼
> 4. 制定执行计划：收集证据 → 申请仲裁 → 准备诉讼
> 5. 预判风险：仲裁时效 1 年，需注意证据保存
>
> � **最终答案**：
> 1. 【协商】与用人单位协商补偿
> 2. 【调解】申请企业劳动争议调解委员会调解
> 3. 【仲裁】向劳动人事争议仲裁委员会申请仲裁（必经程序）
> 4. 【诉讼】对仲裁裁决不服可向法院起诉
>
> 注意事项：保留劳动合同、工资条、辞退通知等证据……

---

## ❓ 常见问题

**Q1：启动时报 `DEEPSEEK_API_KEY` 缺失？**
A：请在 `.env` 中正确填写 Key，注意不要有多余空格。

**Q2：联网搜索失败？**
A：检查 `TAVILY_API_KEY` 是否有效，或临时取消勾选「🌐 联网搜索」。

**Q3：知识库为空？**
A：将法律文档放入 `knowledge_base/` 目录，重启或点击「📥 上传并添加到知识库」。

**Q4：流式输出卡顿？**
A：网络不稳定时可能出现，可切换至非深度思考模式。

**Q5：Embedding 模型下载失败？**
A：国内网络可能超时，在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

**Q6：能否修改默认检索模式？**
A：可以。在 `.env` 中将 `SEARCH_MODE` 修改为 `hybrid`（混合）/ `semantic`（语义）/ `keyword`（关键词）后重启即可。Web 界面默认锁定为 `hybrid` 模式，无需手动选择。

**Q7：能否换用更大的中文 Embedding 模型？**
A：可以。修改 `.env` 中 `EMBEDDING_MODEL_NAME`：
- `BAAI/bge-small-zh-v1.5`（512 维）
- `BAAI/bge-base-zh-v1.5`（768 维）
- `shibing624/text2vec-base-chinese`

---

## ⚠️ 注意事项

1. 🌐 **网络要求**：需要访问 DeepSeek 与 Tavily API
2. 🔐 **密钥安全**：`.env` 文件请勿提交至公开仓库
3. 📚 **知识库维护**：定期更新 `knowledge_base/` 中的法律文档以保持时效性
4. ⚖️ **免责声明**：系统回答仅供参考，具体法律问题请咨询专业律师
5. 💰 **API 费用**：DeepSeek 与 Tavily 均按调用计费，请关注额度

---

## �️ 路线图

- [ ] 引入 ReAct Agent 自动工具调用循环
- [ ] 增加用户登录与多租户隔离
- [ ] 引入 RAGAS 评估指标体系
- [ ] 支持 DeepSeek 自家的 Embedding API（替代本地模型）

---

## 👥 作者
项目维护者名称：ShannZhong
GitHub 仓库地址：https://github.com/ShannZhong/Legal-Consultation-Assistant
问题反馈地址:https://github.com/ShannZhong/Legal-Consultation-Assistant/issues
---

⭐ 如果这个项目对你有帮助，欢迎 Star！
