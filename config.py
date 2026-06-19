from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# ===================== Hugging Face 镜像设置 =====================
# 必须尽早设置，避免 huggingface_hub / transformers 在导入时使用官方端点
# 国内常用镜像：hf-mirror.com（推荐）、aifasthub.com、aliendao.cn
_HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip()
os.environ["HF_ENDPOINT"] = _HF_MIRROR
os.environ["HF_HUB_ENDPOINT"] = _HF_MIRROR  # 新版 hub 变量名
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = _HF_MIRROR  # 兼容旧版
# 禁用外网 telemetry，加快速度
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

# ===================== LLM 配置 =====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedding")

# ===================== 搜索 API 配置 =====================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# ===================== 向量数据库（Chroma）配置 =====================
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "labor_law_kb")

# ===================== Embedding 配置 =====================
# 可选：huggingface / openai / deepseek
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./models_cache")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
# Hugging Face 镜像站点（国内推荐 hf-mirror.com）
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ===================== RAG 配置 =====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
# 检索模式：semantic / keyword / hybrid
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid").strip().lower()
# 混合检索中语义通道的召回数
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "8"))
# 混合检索中关键词通道的召回数
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "8"))

# ===================== 知识库路径 =====================
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
UPLOADS_PATH = os.getenv("UPLOADS_PATH", "./uploads")


# ===================== 配置验证 =====================
def validate_config():
    errors = []
    if not DEEPSEEK_API_KEY:
        errors.append("请在.env文件中配置有效的DEEPSEEK_API_KEY")
    if not TAVILY_API_KEY:
        errors.append("请在.env文件中配置有效的TAVILY_API_KEY")
    return errors


# ===================== 创建目录 =====================
def create_directories():
    os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


class Config:
    pass
