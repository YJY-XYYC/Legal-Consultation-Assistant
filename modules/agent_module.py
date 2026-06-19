from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import config
from modules.tools import get_tools
from collections import deque
from datetime import datetime


class AgentModule:
    def __init__(self, max_history=10):
        self.llm = ChatDeepSeek(
            model=config.DEEPSEEK_CHAT_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            api_base=config.DEEPSEEK_API_BASE,
            temperature=0.2
        )
        self.tools = get_tools()
        self.tool_dict = {tool.name: tool for tool in self.tools}
        self.max_history = max_history
        # 多会话记忆管理
        self.sessions = {}

    def get_session(self, session_id):
        """获取或创建会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": deque(maxlen=self.max_history * 2),
                "created_at": datetime.now().isoformat(),
                "turn_count": 0
            }
        return self.sessions[session_id]

    def add_to_history(self, session_id, task, final_answer, thinking_process=None):
        """添加问答到历史"""
        session = self.get_session(session_id)
        session["history"].append({"role": "user", "content": task, "time": datetime.now().isoformat()})
        if thinking_process:
            session["history"].append({"role": "thinking", "content": thinking_process, "time": datetime.now().isoformat()})
        session["history"].append({"role": "assistant", "content": final_answer, "time": datetime.now().isoformat()})
        session["turn_count"] += 1

    def get_chat_history_str(self, session_id):
        """获取格式化的对话历史"""
        session = self.get_session(session_id)
        if not session["history"]:
            return "（暂无对话历史）"
        lines = []
        for msg in session["history"]:
            if msg["role"] == "user":
                lines.append(f"用户：{msg['content']}")
            elif msg["role"] == "assistant":
                lines.append(f"助手：{msg['content']}")
            elif msg["role"] == "thinking":
                lines.append(f"助手（思考）：{msg['content']}")
        return "\n".join(lines)

    def get_history_messages(self, session_id):
        """获取历史消息列表"""
        session = self.get_session(session_id)
        return list(session["history"])

    def clear_history(self, session_id):
        """清空指定会话历史"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].clear()
            self.sessions[session_id]["turn_count"] = 0
            return True
        return False

    def get_session_info(self, session_id):
        """获取会话信息"""
        session = self.get_session(session_id)
        return {
            "turn_count": session["turn_count"],
            "history_count": len(session["history"]),
            "created_at": session["created_at"]
        }

    def run_task(self, task, deep_thinking=False):
        try:
            if deep_thinking:
                # 深度思考模式：先输出思考过程，再输出最终答案
                # 第一步：生成思考过程
                thinking_prompt = f"""
                你是一个专业的劳动与社会保障法律智能思考助手。请对以下任务进行深度分析和推理：

                用户任务：{task}

                请按照以下步骤进行思考：
                1. 明确任务目标和要求
                2. 分析相关的劳动法或社会保障法规定
                3. 分析可能的解决方法
                4. 评估每种方法的优缺点
                5. 制定详细的执行计划
                6. 预判可能遇到的问题和解决方案

                请用中文输出详细的思考过程，使用 markdown 格式。
                """
                
                thinking_response = self.llm.invoke([HumanMessage(content=thinking_prompt)])
                thinking_process = thinking_response.content

                # 第二步：生成最终答案
                answer_prompt = f"""
                基于以下思考过程，给出最终答案：

                思考过程：
                {thinking_process}

                用户任务：{task}

                请根据上述思考过程，给出清晰、简洁、准确的最终答案。
                """
                
                answer_response = self.llm.invoke([HumanMessage(content=answer_prompt)])
                final_answer = answer_response.content

                return {
                    "thinking_process": thinking_process,
                    "final_answer": final_answer
                }
            else:
                # 普通模式：直接回答
                response = self.llm.invoke([
                    HumanMessage(content=f"""
                    你是一个专业的劳动与社会保障法律智能助手，可以直接回答问题，也可以使用工具。
                    可用工具：{[t.name for t in self.tools]}
                    用户任务：{task}
                    请直接给出最终答案。
                    """)
                ])
                return {"thinking_process": None, "final_answer": response.content}
        
        except Exception as e:
            return {"thinking_process": None, "final_answer": f"执行失败：{str(e)}"}

    def run_task_stream(self, task, deep_thinking=False, enable_web_search=False, session_id="default"):
        """流式输出版本，实时返回思考过程和最终答案"""
        try:
            chat_history = self.get_chat_history_str(session_id)

            if deep_thinking:
                # 第一步：流式生成思考过程
                web_hint = "你可以使用 web_search 工具进行联网搜索获取最新信息。" if enable_web_search else ""
                thinking_prompt = f"""
                你是一个专业的劳动与社会保障法律智能思考助手。请对以下任务进行深度分析和推理。

                对话历史：
                {chat_history}

                用户任务：{task}
                {web_hint}

                请按照以下步骤进行思考：
                1. 明确任务目标和要求
                2. 分析相关的劳动法或社会保障法规定
                3. 分析可能的解决方法
                4. 评估每种方法的优缺点
                5. 制定详细的执行计划
                6. 预判可能遇到的问题和解决方案

                请用中文输出详细的思考过程，使用 markdown 格式。
                """

                thinking_process = ""
                for chunk in self.llm.stream([HumanMessage(content=thinking_prompt)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        thinking_process += chunk.content
                        yield {"type": "thinking", "content": thinking_process, "delta": chunk.content}

                yield {"type": "thinking_complete", "content": thinking_process}

                # 第二步：如果需要联网搜索，先执行搜索
                web_context = ""
                if enable_web_search:
                    try:
                        api_wrapper = TavilySearchAPIWrapper()
                        raw_results = api_wrapper.raw_results(
                            task,
                            max_results=5,
                            include_answer=False,
                        )
                        web_results = raw_results.get("results", [])

                        # 格式化搜索结果，显示标题和链接
                        if isinstance(web_results, list):
                            links_text = "\n".join([f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)])
                        else:
                            links_text = str(web_results)

                        web_context = f"\n\n联网搜索结果：\n{links_text}"
                        yield {"type": "web_search", "content": links_text}
                    except Exception as e:
                        web_context = f"\n\n联网搜索失败：{str(e)}"
                        yield {"type": "web_search_error", "content": str(e)}

                # 第三步：流式生成最终答案
                answer_prompt = f"""
                基于以下思考过程{web_context}，结合对话历史，给出最终答案：

                对话历史：
                {chat_history}

                思考过程：
                {thinking_process}

                用户任务：{task}

                请根据上述思考过程和对话历史，给出清晰、简洁、准确的最终答案。
                """

                final_answer = ""
                for chunk in self.llm.stream([HumanMessage(content=answer_prompt)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        final_answer += chunk.content
                        yield {"type": "answer", "content": final_answer, "delta": chunk.content}

                yield {"type": "answer_complete", "content": final_answer}
                self.add_to_history(session_id, task, final_answer, thinking_process)

            else:
                # 普通模式：流式直接回答
                web_context = ""

                # 如果需要联网搜索，先执行搜索
                if enable_web_search:
                    try:
                        api_wrapper = TavilySearchAPIWrapper()
                        raw_results = api_wrapper.raw_results(
                            task,
                            max_results=5,
                            include_answer=False,
                        )
                        web_results = raw_results.get("results", [])

                        # 格式化搜索结果，显示标题和链接
                        if isinstance(web_results, list):
                            links_text = "\n".join([f"{i+1}. {r.get('title', '无标题')}\n   {r.get('url', '')}" for i, r in enumerate(web_results)])
                        else:
                            links_text = str(web_results)

                        web_context = f"\n\n联网搜索结果：\n{links_text}"
                        yield {"type": "web_search", "content": links_text}
                    except Exception as e:
                        web_context = f"\n\n联网搜索失败：{str(e)}"
                        yield {"type": "web_search_error", "content": str(e)}

                response_content = ""
                for chunk in self.llm.stream([
                    HumanMessage(content=f"""
                    你是一个专业的劳动与社会保障法律智能助手，可以直接回答问题，也可以使用工具。
                    可用工具：{[t.name for t in self.tools]}
                    {web_context}

                    对话历史：
                    {chat_history}

                    用户任务：{task}
                    请结合对话历史和上述联网搜索结果，给出清晰、简洁、准确的最终答案。
                    """)
                ]):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        yield {"type": "answer", "content": response_content, "delta": chunk.content}

                yield {"type": "answer_complete", "content": response_content}
                self.add_to_history(session_id, task, response_content)

        except Exception as e:
            yield {"type": "error", "content": f"执行失败：{str(e)}"}