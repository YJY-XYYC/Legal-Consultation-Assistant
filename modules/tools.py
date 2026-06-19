from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
# 修复：从 langchain_experimental 导入 PythonREPL（官方要求）
from langchain_experimental.utilities.python import PythonREPL
import sympy as sp

# 1. 代码执行工具
@tool
def python_runner(code: str) -> str:
    """用于执行 Python 代码，返回执行结果。输入必须是合法的 Python 代码字符串。"""
    try:
        repl = PythonREPL()
        result = repl.run(code)
        return f"代码执行结果：\n{result}"
    except Exception as e:
        return f"执行失败：{str(e)}"

# 2. 计算器工具
@tool
def calculator(expression: str) -> str:
    """用于计算数学表达式，支持加减乘除、平方、开方、三角函数等。输入必须是数学表达式字符串。"""
    try:
        result = sp.sympify(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"

# 3. 网页搜索工具
@tool
def web_search(query: str) -> str:
    """用于联网搜索实时信息、最新知识。输入是搜索关键词。"""
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return f"搜索结果：\n{result}"
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 获取工具列表
def get_tools():
    return [python_runner, calculator, web_search]