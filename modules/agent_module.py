from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import config
from modules.tools import get_tools
from collections import deque
from datetime import datetime


AGENT_SYSTEM_PROMPT = """你是一个专业的劳动与社会保障法律智能助手，名字叫"法律小智"。

你的能力包括：
- knowledge_search: 在劳动法知识库中检索相关法律条文、规定和实务信息
- legal_calculator: 计算劳动法相关赔偿（经济补偿金N、违法解除赔偿金2N、加班费、未休年假补偿）
- web_search: 联网搜索最新的法律法规和政策动态（如果启用了联网搜索）

回答规则：
1. 优先使用知识库中的信息，回答时注明信息来源文件
2. 如果知识库信息不足，可使用联网搜索补充最新信息
3. 涉及赔偿金额计算时，先检索相关规定确认适用条款，再使用计算器得出结果
4. 回答要专业、准确、清晰，引用相关法律条文编号
5. 对于不确定的问题，如实说明，不要编造法律条文
6. 如果用户问题不涉及劳动与社会保障法，礼貌引导到相关话题
7. 使用中文回答
"""

AGENT_DEEP_THINKING_PROMPT = """你是一个专业的劳动与社会保障法律深度分析助手，名字叫"法律小智"。

你的能力包括：
- knowledge_search: 在劳动法知识库中检索相关法律条文、规定和实务信息
- legal_calculator: 计算劳动法相关赔偿（经济补偿金N、违法解除赔偿金2N、加班费、未休年假补偿）
- web_search: 联网搜索最新的法律法规和政策动态（如果启用了联网搜索）

你必须按照以下深度分析流程工作：
1. **明确问题**：先用自己的话重述用户的核心诉求，界定问题的法律性质
2. **检索法条**：调用 knowledge_search 检索所有相关法律条文，不要只查一次，要从不同角度多次检索
3. **交叉验证**：如果知识库信息不够全面或可能存在时效性问题，调用 web_search 获取最新政策
4. **精确计算**：涉及金额的，必须调用 legal_calculator 进行精确计算，并展示计算过程
5. **综合分析**：综合所有检索和计算结果，给出完整的法律分析，包括：
   - 适用的法律条文及编号
   - 权利与义务分析
   - 可能的救济途径
   - 风险评估与建议
6. **引用来源**：所有结论必须注明来自知识库哪个文件或联网搜索的哪个来源

回答要结构清晰，使用 markdown 格式，分章节组织。对于复杂问题，用表格对比不同情形。使用中文回答。
"""

THINKING_PROMPT = """你是一个法律分析助手。请对以下用户问题进行深度分析思考，输出你的思考过程：

1. **理解诉求**：用自己的话重述用户的核心诉求和法律问题
2. **定位领域**：识别涉及的法律领域（劳动合同、社会保险、劳动争议、工资福利等）
3. **检索策略**：列出需要查询的关键法律条文和检索方向
4. **分析框架**：规划分析的维度和步骤

注意：你只需要输出思考过程和分析框架，不要给出最终答案。后续会有专门的检索和计算步骤来完成回答。"""


class AgentModule:
    def __init__(self, rag_module=None, max_history=10):
        self.rag_module = rag_module
        self.llm = ChatDeepSeek(
            model=config.DEEPSEEK_CHAT_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            api_base=config.DEEPSEEK_API_BASE,
            temperature=0.2,
            max_tokens=4096,
        )
        self.max_history = max_history
        self.sessions = {}

    # ======================== 会话管理 ========================

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": deque(maxlen=self.max_history * 2),
                "created_at": datetime.now().isoformat(),
                "turn_count": 0
            }
        return self.sessions[session_id]

    def add_to_history(self, session_id, task, final_answer):
        session = self.get_session(session_id)
        session["history"].append({"role": "user", "content": task, "time": datetime.now().isoformat()})
        session["history"].append({"role": "assistant", "content": final_answer, "time": datetime.now().isoformat()})
        session["turn_count"] += 1

    def get_chat_history_str(self, session_id):
        session = self.get_session(session_id)
        if not session["history"]:
            return "（暂无对话历史）"
        lines = []
        for msg in session["history"]:
            if msg["role"] == "user":
                lines.append(f"用户：{msg['content']}")
            elif msg["role"] == "assistant":
                lines.append(f"助手：{msg['content']}")
        return "\n".join(lines)

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

    # ======================== Agent 构建 ========================

    def _build_chat_history_messages(self, session_id):
        """将会话历史转换为 LangChain Message 格式。"""
        session = self.get_session(session_id)
        messages = []
        for msg in session["history"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _build_agent(self, tools, deep_thinking=False):
        """构建 LangGraph ReAct Agent。"""
        system_prompt = AGENT_DEEP_THINKING_PROMPT if deep_thinking else AGENT_SYSTEM_PROMPT
        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    # ======================== 流式执行 ========================

    def _generate_thinking(self, task):
        """Phase 1: 流式生成深度思考分析文本。"""
        thinking_messages = [
            SystemMessage(content=THINKING_PROMPT),
            HumanMessage(content=f"用户问题：{task}")
        ]
        thinking_text = ""
        for chunk in self.llm.stream(thinking_messages):
            if chunk.content:
                thinking_text += chunk.content
                yield thinking_text
        yield thinking_text  # 最终完整文本

    def run_task_stream(self, task, enable_web_search=False, deep_thinking=False, session_id="default"):
        """流式执行 Agent 任务。

        深度思考模式分两阶段：
        Phase 1 - 调用 LLM 生成结构化分析文本（始终可见）
        Phase 2 - 执行 ReAct Agent（可能调用工具）
        """
        # ===== Phase 1: 深度思考分析 =====
        if deep_thinking:
            try:
                for thinking_text in self._generate_thinking(task):
                    yield {"type": "thinking", "content": thinking_text}
                yield {"type": "thinking_complete", "content": thinking_text}
            except Exception as e:
                yield {"type": "thinking", "content": f"*（思考生成异常：{e}）*"}
                yield {"type": "thinking_complete", "content": f"*（思考生成异常：{e}）*"}

        # ===== Phase 2: Agent 执行 =====
        tools = get_tools(self.rag_module, enable_web_search)
        agent = self._build_agent(tools, deep_thinking=deep_thinking)
        chat_history = self._build_chat_history_messages(session_id)

        # 将当前问题追加到消息列表
        input_messages = chat_history + [HumanMessage(content=task)]

        try:
            final_answer = ""
            current_msg_id = None
            streaming_answer = True  # 默认假设是答案，遇到工具调用再关闭

            for chunk in agent.stream(
                {"messages": input_messages},
                stream_mode=["updates", "messages"],
            ):
                mode, data = chunk[0], chunk[1]

                # ---- token 级流式：逐字输出最终答案 ----
                if mode == "messages":
                    msg, metadata = data
                    if metadata.get("langgraph_node") == "model":
                        # 检测新消息：重置 streaming 状态
                        msg_id = getattr(msg, 'id', None)
                        if msg_id and msg_id != current_msg_id:
                            current_msg_id = msg_id
                            streaming_answer = True
                        # 如果包含工具调用 token，标记非答案消息
                        if hasattr(msg, 'tool_call_chunks') and msg.tool_call_chunks:
                            streaming_answer = False
                        # 纯文本 token，且确认为最终答案
                        elif hasattr(msg, 'content') and msg.content and streaming_answer:
                            final_answer += str(msg.content)
                            yield {"type": "answer_chunk", "content": final_answer}

                # ---- 节点级：工具调用与完成事件 ----
                elif mode == "updates":
                    # 模型节点输出：可能包含工具调用决策或最终回答
                    if "model" in data:
                        messages = data["model"].get("messages", [])
                        for msg in messages:
                            # 检查是否包含工具调用
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = tc.get("name", "unknown")
                                    tool_args = tc.get("args", {})
                                    yield {
                                        "type": "tool_start",
                                        "tool": tool_name,
                                        "input": str(tool_args),
                                    }
                            # 纯文本内容（不含工具调用）= 最终回答
                            elif msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                                if not final_answer:
                                    final_answer = str(msg.content)

                    # 工具节点输出：包含工具执行结果
                    if "tools" in data:
                        messages = data["tools"].get("messages", [])
                        for msg in messages:
                            if isinstance(msg, ToolMessage):
                                yield {
                                    "type": "tool_end",
                                    "tool": getattr(msg, 'name', 'unknown'),
                                    "output": str(msg.content) if msg.content else "",
                                }

            if final_answer:
                yield {"type": "answer", "content": final_answer}
                yield {"type": "answer_complete", "content": final_answer}
                self.add_to_history(session_id, task, final_answer)
            else:
                yield {"type": "error", "content": "Agent 未能生成有效回答，请重试。"}

        except Exception as e:
            yield {"type": "error", "content": f"执行失败：{str(e)}"}
