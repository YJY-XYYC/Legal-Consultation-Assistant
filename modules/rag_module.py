from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import os
import re
import shutil
import hashlib
import jieba
import config
from collections import deque
from datetime import datetime


class RAGModule:
    def __init__(self, max_history=10, search_mode=None):
        self.llm = ChatDeepSeek(
            model=config.DEEPSEEK_CHAT_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            api_base=config.DEEPSEEK_API_BASE,
            temperature=0.1,
            max_tokens=2048
        )
        self.documents = []
        self.max_history = max_history
        self.sessions = {}
        self.search_mode = (search_mode or config.SEARCH_MODE or "hybrid").lower()
        if self.search_mode not in ("semantic", "keyword", "hybrid"):
            self.search_mode = "hybrid"

        # Embedding & 向量库相关状态
        self.embeddings = None
        self.vectorstore = None
        self.embedding_error = None
        self._embeddings_initialized = False
        self._chroma_loaded_from_disk = False

        # 中文常用停用词（关键词检索用）
        self.stopwords = set([
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '们',
            '这', '那', '有', '没有', '和', '与', '或', '但', '因为', '所以',
            '如果', '虽然', '然而', '而且', '并且', '或者', '以及', '关于', '对于',
            '根据', '按照', '通过', '经过', '由于', '为了', '其中', '包括', '在内',
            '之一', '什么', '怎么', '如何', '为什么', '多少', '几个', '哪个', '哪些',
            '这个', '那个', '一个', '一些', '可以', '能够', '会', '能', '要', '应该',
            '必须', '得', '被', '把', '将', '向', '对', '于', '从', '到', '为',
            '及', '等'
        ])

        # ============ Prompt 模板 ============
        self.prompt = ChatPromptTemplate.from_template("""
你是一个专业的劳动与社会保障法律助手。请根据以下参考资料和对话历史回答用户的问题。
请仔细阅读参考资料中的内容，如果其中包含与问题相关的信息，请基于这些内容进行回答。
如果参考资料中的信息不足以完整回答问题，请基于现有信息尽量回答，并在回答时说明这一点。
只有当参考资料中完全没有相关信息时，才回答"根据现有知识库无法回答该问题"。
请用中文详细回答问题。

对话历史：
{chat_history}

参考资料：
{context}

用户问题：{question}
""")

        self.web_prompt = ChatPromptTemplate.from_template("""
你是一个专业的劳动与社会保障法律助手。请综合以下知识库内容、联网搜索结果和对话历史回答用户的问题。
请优先使用知识库中的信息，如果知识库信息不足，可以参考联网搜索结果进行补充。
请用中文详细回答问题，并在回答中注明来源是来自知识库还是联网搜索。

对话历史：
{chat_history}

知识库内容：
{kb_context}

联网搜索结果：
{web_context}

用户问题：{question}
""")

    # ======================== Embedding 初始化 ========================
    def _init_embeddings(self):
        """延迟初始化 Embedding 模型，加载失败时降级为纯关键词检索。"""
        if self._embeddings_initialized:
            return
        self._embeddings_initialized = True

        if self.search_mode == "keyword":
            return

        provider = (config.EMBEDDING_PROVIDER or "huggingface").lower()
        try:
            if provider == "huggingface":
                # 强制覆盖，确保使用国内镜像（config.py 中已设置，这里再次兜底）
                os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
                os.environ["HF_HUB_ENDPOINT"] = config.HF_ENDPOINT
                os.environ["HUGGINGFACE_HUB_ENDPOINT"] = config.HF_ENDPOINT
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL_NAME,
                    cache_folder=config.EMBEDDING_CACHE_DIR,
                    model_kwargs={"device": config.EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:
                raise ValueError(f"暂不支持的 Embedding 提供方: {provider}")
        except Exception as e:
            self.embedding_error = f"Embedding 加载失败: {e}"
            print(f"[RAG] {self.embedding_error}，将自动降级为关键词检索。")
            self.embeddings = None
            if self.search_mode in ("semantic", "hybrid"):
                print("[RAG] 当前 search_mode 已自动降级为 keyword。")
                self.search_mode = "keyword"

    # ======================== 会话记忆管理 ========================
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": deque(maxlen=self.max_history * 2),
                "created_at": datetime.now().isoformat(),
                "turn_count": 0
            }
        return self.sessions[session_id]

    def add_to_history(self, session_id, question, answer):
        session = self.get_session(session_id)
        session["history"].append({"role": "user", "content": question, "time": datetime.now().isoformat()})
        session["history"].append({"role": "assistant", "content": answer, "time": datetime.now().isoformat()})
        session["turn_count"] += 1

    def get_chat_history_str(self, session_id):
        session = self.get_session(session_id)
        if not session["history"]:
            return "（暂无对话历史）"
        history_lines = []
        for msg in session["history"]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}：{msg['content']}")
        return "\n".join(history_lines)

    def get_history_messages(self, session_id):
        session = self.get_session(session_id)
        return list(session["history"])

    def clear_history(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].clear()
            self.sessions[session_id]["turn_count"] = 0
            return True
        return False

    def get_session_info(self, session_id):
        session = self.get_session(session_id)
        return {
            "turn_count": session["turn_count"],
            "history_count": len(session["history"]),
            "created_at": session["created_at"]
        }

    # ======================== 知识库加载 ========================
    def load_knowledge_base(self, folder_path=None):
        folder_path = folder_path or config.KNOWLEDGE_BASE_PATH
        self.documents = []
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith(".txt") or filename.endswith(".md"):
                    loader = TextLoader(file_path, encoding="utf-8")
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                self.documents.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {filename} 失败：{e}")

        if not self.documents:
            print("警告：未加载任何知识库文档，请在 knowledge_base 文件夹中放入文档！")
            # 知识库为空时，清理旧的向量库数据
            self._clean_vectorstore()
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(self.documents)
        self.documents = splits

        # 写入 / 更新 Chroma 向量库（始终检查文件变化，不依赖当前 search_mode）
        try:
            if self._should_rebuild_vectorstore(folder_path):
                self._init_embeddings()
                if self.embeddings is not None:
                    self._build_vectorstore(splits, folder_path)
                else:
                    print("[RAG] Embedding 模型未就绪，跳过向量库重建。")
                    self.vectorstore = None
                    self._chroma_loaded_from_disk = False
            else:
                print("[RAG] 知识库文件未变化，跳过向量库重建。")
                self.vectorstore = None  # 延迟到首次检索时从磁盘加载
                self._chroma_loaded_from_disk = False
        except Exception as e:
            print(f"[RAG] 构建向量库失败: {e}，将降级为关键词检索。")
            self.search_mode = "keyword"

        return len(splits)

    def _build_vectorstore(self, splits, folder_path=None):
        """构建/重建 Chroma 向量库。已存在则先清空，再灌入新数据。"""
        from langchain_chroma import Chroma

        persist_dir = config.CHROMA_PERSIST_DIRECTORY
        collection = config.COLLECTION_NAME

        if self.embeddings is None:
            return

        # 先释放旧向量库引用，避免 Windows 文件锁定导致 shutil.rmtree 失败
        self.vectorstore = None
        self._chroma_loaded_from_disk = False

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            shutil.rmtree(persist_dir)
            os.makedirs(persist_dir, exist_ok=True)

        def _relevance_from_cosine(distance: float) -> float:
            return max(0.0, min(1.0, 1.0 - distance))

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=collection,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=_relevance_from_cosine,
        )
        try:
            self.vectorstore.persist()
        except Exception:
            pass
        self._chroma_loaded_from_disk = True

        # 保存知识库文件指纹，用于后续判断是否需要重建
        if folder_path:
            self._save_fingerprint(folder_path)

    # ======================== 向量库清理 ========================
    def _clean_vectorstore(self):
        """清空向量库（释放内存引用并删除磁盘数据）。"""
        self.vectorstore = None
        self._chroma_loaded_from_disk = False
        persist_dir = config.CHROMA_PERSIST_DIRECTORY
        if os.path.isdir(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
            except Exception as e:
                print(f"[RAG] 清理向量库目录失败: {e}")

    # ======================== 知识库指纹（增量更新判断） ========================
    def _compute_kb_fingerprint(self, folder_path):
        """计算知识库文件指纹（基于文件路径、修改时间和分块参数）。"""
        hasher = hashlib.md5()
        if not os.path.isdir(folder_path):
            return None
        files = sorted(os.listdir(folder_path))
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                hasher.update(fname.encode("utf-8"))
                hasher.update(str(os.path.getmtime(fpath)).encode("utf-8"))
        hasher.update(str(config.CHUNK_SIZE).encode("utf-8"))
        hasher.update(str(config.CHUNK_OVERLAP).encode("utf-8"))
        return hasher.hexdigest()

    def _fingerprint_path(self, folder_path):
        """指纹文件存储路径。"""
        return os.path.join(config.CHROMA_PERSIST_DIRECTORY, ".kb_fingerprint")

    def _load_fingerprint(self, folder_path):
        """从磁盘读取上次保存的指纹。"""
        fp_file = self._fingerprint_path(folder_path)
        if not os.path.isfile(fp_file):
            return None
        with open(fp_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _save_fingerprint(self, folder_path):
        """将当前指纹写入磁盘。"""
        fp = self._compute_kb_fingerprint(folder_path)
        if fp is None:
            return
        fp_file = self._fingerprint_path(folder_path)
        os.makedirs(os.path.dirname(fp_file), exist_ok=True)
        with open(fp_file, "w", encoding="utf-8") as f:
            f.write(fp)

    def _should_rebuild_vectorstore(self, folder_path):
        """判断是否需要重建向量库。"""
        persist_dir = config.CHROMA_PERSIST_DIRECTORY
        # 向量库目录不存在或为空 -> 需要重建
        if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
            return True
        # 指纹文件不存在（首次升级或异常丢失）-> 需要重建
        if not os.path.isfile(self._fingerprint_path(folder_path)):
            return True
        current_fp = self._compute_kb_fingerprint(folder_path)
        stored_fp = self._load_fingerprint(folder_path)
        if current_fp is None or stored_fp is None:
            return True
        return current_fp != stored_fp

    # ======================== 关键词检索（保留兼容） ========================
    def _extract_keywords(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text).lower()
        # 先用空格初步切分，区分英文和中文字段
        segments = text.split()
        keywords = []
        for seg in segments:
            if seg.isascii():
                # 英文/数字段：直接按原逻辑保留
                if len(seg) >= 2 and seg not in self.stopwords:
                    keywords.append(seg)
            else:
                # 中文段：用 jieba 精准分词
                for token in jieba.cut(seg):
                    token = token.strip().lower()
                    if len(token) >= 2 and token not in self.stopwords:
                        keywords.append(token)
        return keywords

    def _calculate_relevance(self, doc_content, keywords):
        content_lower = doc_content.lower()
        score = 0
        matched_keywords = []
        for keyword in keywords:
            if keyword in content_lower:
                score += 3
                matched_keywords.append(keyword)
            else:
                # 完整关键词未命中时，用2字滑动窗口做子串匹配（中文 bigram）
                for i in range(len(keyword) - 1):
                    bigram = keyword[i:i+2]
                    if bigram in content_lower:
                        score += 1
        common_chars = set(keywords) & set(content_lower)
        score += len(common_chars) * 0.5
        return score, matched_keywords

    def _keyword_search(self, question, top_k=None):
        if not self.documents:
            return []
        top_k = top_k or config.KEYWORD_TOP_K
        keywords = self._extract_keywords(question)
        if not keywords:
            return self.documents[:top_k]

        scored_docs = []
        for doc in self.documents:
            score, matched = self._calculate_relevance(doc.page_content, keywords)
            if score > 0 or matched:
                scored_docs.append((score, len(matched), doc))
        scored_docs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if not scored_docs:
            return self.documents[:top_k]
        return [doc for _, _, doc in scored_docs[:top_k]]

    # ======================== 语义检索 ========================
    def _semantic_search(self, question, top_k=None):
        """使用 Chroma + Embedding 进行向量召回。"""
        if self.vectorstore is None:
            try:
                self._init_embeddings()
                if self.embeddings is not None:
                    from langchain_chroma import Chroma

                    def _relevance_from_cosine(distance: float) -> float:
                        return max(0.0, min(1.0, 1.0 - distance))

                    self.vectorstore = Chroma(
                        collection_name=config.COLLECTION_NAME,
                        embedding_function=self.embeddings,
                        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
                        collection_metadata={"hnsw:space": "cosine"},
                        relevance_score_fn=_relevance_from_cosine,
                    )
                    self._chroma_loaded_from_disk = True
            except Exception as e:
                print(f"[RAG] 向量库加载失败: {e}，降级为关键词检索。")
                self.vectorstore = None

        if self.vectorstore is None:
            return []

        top_k = top_k or config.SEMANTIC_TOP_K
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=top_k
            )
            return [doc for doc, _ in results]
        except Exception as e:
            print(f"[RAG] 语义检索失败: {e}")
            return []

    # ======================== 混合检索（RRF 融合） ========================
    def _hybrid_search(self, question, top_k=None):
        """
        使用 Reciprocal Rank Fusion 融合关键词 + 语义结果。
        RRF 公式：score(d) = Σ 1 / (k + rank_i)
        """
        k = top_k or config.TOP_K
        if not hasattr(self, "_doc_index"):
            self._doc_index = {}

        sem_docs = self._semantic_search(question, top_k=config.SEMANTIC_TOP_K)
        kw_docs = self._keyword_search(question, top_k=config.KEYWORD_TOP_K)

        rrf_k = 60
        scores = {}
        order = []

        for rank, doc in enumerate(sem_docs, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in order:
                order.append(key)
            self._doc_index[key] = doc

        for rank, doc in enumerate(kw_docs, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in order:
                order.append(key)
            self._doc_index[key] = doc

        sorted_keys = sorted(order, key=lambda x: scores[x], reverse=True)
        return [self._doc_index[key] for key in sorted_keys[:k]]

    # ======================== 路由检索 ========================
    def _retrieve(self, question, top_k=None):
        if self.search_mode == "semantic":
            if self.embeddings is not None:
                docs = self._semantic_search(question, top_k=top_k or config.SEMANTIC_TOP_K)
            else:
                # Embedding 未就绪，降级为关键词检索
                docs = self._keyword_search(question, top_k=top_k or config.KEYWORD_TOP_K)
        elif self.search_mode == "keyword":
            docs = self._keyword_search(question, top_k=top_k or config.KEYWORD_TOP_K)
        else:
            docs = self._hybrid_search(question, top_k=top_k or config.TOP_K)

        if not docs and self.documents:
            docs = self.documents[: top_k or config.TOP_K]
        return docs

    def _format_docs(self, docs):
        result = []
        for doc in docs:
            source = doc.metadata.get('source', '未知')
            content = doc.page_content.strip()
            result.append(f"【{source}】\n{content}")
        return "\n\n".join(result)

    # ======================== 公开检索方法（供 Agent 工具调用） ========================
    def search(self, query, top_k=None, search_mode=None):
        """公开检索方法：仅检索知识库，不修改对话历史，不调用 LLM。
        返回 (formatted_text, list_of_docs) 二元组。
        """
        original_mode = self.search_mode
        if search_mode:
            self.search_mode = search_mode
        try:
            docs = self._retrieve(query, top_k=top_k)
            if not docs and self.documents:
                docs = self.documents[: top_k or config.TOP_K]
            formatted = self._format_docs(docs) if docs else ""
            return formatted, docs
        finally:
            self.search_mode = original_mode

    # ======================== 对外问答 ========================
    def ask(self, question, session_id="default", search_mode=None):
        if search_mode:
            self.search_mode = search_mode
        if not self.documents:
            return "请先加载知识库！"

        relevant_docs = self._retrieve(question)
        if not relevant_docs:
            relevant_docs = self.documents[:config.TOP_K]

        context = self._format_docs(relevant_docs)
        chat_history = self.get_chat_history_str(session_id)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history
        })

        self.add_to_history(session_id, question, answer)
        return answer

    def ask_stream(self, question, session_id="default", search_mode=None):
        if search_mode:
            self.search_mode = search_mode
        if not self.documents:
            yield "请先加载知识库！"
            return

        relevant_docs = self._retrieve(question)
        if not relevant_docs:
            relevant_docs = self.documents[:config.TOP_K]

        context = self._format_docs(relevant_docs)
        chat_history = self.get_chat_history_str(session_id)

        messages = self.prompt.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history
        })
        full_answer = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content

        self.add_to_history(session_id, question, full_answer)

    def ask_with_web_search(self, question, session_id="default", search_mode=None):
        if search_mode:
            self.search_mode = search_mode

        relevant_docs = self._retrieve(question)
        if not relevant_docs:
            relevant_docs = self.documents[:config.TOP_K]
        kb_context = self._format_docs(relevant_docs)
        chat_history = self.get_chat_history_str(session_id)

        web_results = []
        web_error = None
        try:
            api_wrapper = TavilySearchAPIWrapper()
            raw_results = api_wrapper.raw_results(
                question,
                max_results=5,
                include_answer=False,
            )
            web_results = raw_results.get("results", []) or []
            if isinstance(web_results, list) and web_results:
                web_context = "\n".join(
                    [f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)]
                )
            else:
                web_context = "（联网搜索未返回结果）"
        except Exception as e:
            web_error = str(e)
            web_context = f"联网搜索失败：{web_error}"

        chain = self.web_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "kb_context": kb_context,
            "web_context": web_context,
            "question": question,
            "chat_history": chat_history
        })

        self.add_to_history(session_id, question, answer)
        return {
            "answer": answer,
            "web_results": web_results,
            "web_error": web_error,
        }

    def ask_with_web_search_stream(self, question, session_id="default", search_mode=None):
        if search_mode:
            self.search_mode = search_mode

        relevant_docs = self._retrieve(question)
        if not relevant_docs:
            relevant_docs = self.documents[:config.TOP_K]
        kb_context = self._format_docs(relevant_docs)
        chat_history = self.get_chat_history_str(session_id)

        web_results = []
        web_error = None
        try:
            api_wrapper = TavilySearchAPIWrapper()
            raw_results = api_wrapper.raw_results(
                question,
                max_results=5,
                include_answer=False,
            )
            web_results = raw_results.get("results", []) or []
            if isinstance(web_results, list) and web_results:
                web_context = "\n".join(
                    [f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)]
                )
            else:
                web_context = "（联网搜索未返回结果）"
        except Exception as e:
            web_error = str(e)
            web_context = f"联网搜索失败：{web_error}"

        web_context_dict = {
            "web_results": web_results,
            "web_error": web_error,
        }

        messages = self.web_prompt.invoke({
            "kb_context": kb_context,
            "web_context": web_context,
            "question": question,
            "chat_history": chat_history
        })

        def stream_generator():
            full_answer = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield chunk.content
            self.add_to_history(session_id, question, full_answer)

        return stream_generator(), web_context_dict

    def get_search_mode(self):
        return self.search_mode
