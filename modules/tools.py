from langchain_core.tools import tool, StructuredTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from typing import Optional
from pydantic import BaseModel, Field


# ======================== 知识库检索工具 ========================

def _make_knowledge_search(rag_module):
    """工厂函数：创建绑定到指定 RAGModule 实例的 knowledge_search 工具。"""

    @tool
    def knowledge_search(query: str) -> str:
        """在劳动与社会保障法知识库中检索相关法律条文和规定。
        输入应为清晰的法律问题或关键词，例如 "加班费计算标准" 或 "违法解除劳动合同赔偿"。
        返回相关的知识库文档片段及其来源文件名。
        """
        if rag_module is None:
            return "知识库不可用，请先加载文档。"
        formatted, docs = rag_module.search(query, top_k=5)
        if not formatted:
            return "未在知识库中找到相关内容。"
        return formatted

    return knowledge_search


# ======================== 联网搜索工具 ========================

@tool
def web_search(query: str) -> str:
    """联网搜索最新的法律法规、政策动态和实务信息。
    用于补充知识库中可能不包含的最新信息。
    输入应为搜索关键词或问题。
    """
    try:
        api_wrapper = TavilySearchAPIWrapper()
        raw_results = api_wrapper.raw_results(
            query,
            max_results=5,
            include_answer=False,
        )
        results = raw_results.get("results", []) or []
        if not results:
            return "联网搜索未返回结果。"
        lines = []
        for i, r in enumerate(results):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"{i+1}. {title}\n   链接: {url}\n   摘要: {content}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"联网搜索失败：{str(e)}"


# ======================== 赔偿计算器工具 ========================

class LegalCalculatorInput(BaseModel):
    """赔偿计算器的输入参数"""
    calc_type: str = Field(
        description="计算类型：N（经济补偿金）、2N（违法解除赔偿金）、overtime（加班费）、annual_leave（未休年假补偿）"
    )
    years: Optional[float] = Field(
        default=None,
        description="工作年限，N/2N 类型必填，不满1年按比例"
    )
    monthly_salary: Optional[float] = Field(
        default=None,
        description="月平均工资（元），计算基数"
    )
    overtime_hours_weekday: Optional[float] = Field(
        default=0,
        description="工作日延长工作时间（小时），用于 overtime 类型"
    )
    overtime_hours_weekend: Optional[float] = Field(
        default=0,
        description="休息日加班时间（小时），用于 overtime 类型"
    )
    overtime_hours_holiday: Optional[float] = Field(
        default=0,
        description="法定节假日加班时间（小时），用于 overtime 类型"
    )
    unused_leave_days: Optional[float] = Field(
        default=None,
        description="未休年假天数，用于 annual_leave 类型"
    )


def _legal_calculator_impl(
    calc_type: str,
    years: Optional[float] = None,
    monthly_salary: Optional[float] = None,
    overtime_hours_weekday: float = 0,
    overtime_hours_weekend: float = 0,
    overtime_hours_holiday: float = 0,
    unused_leave_days: Optional[float] = None,
) -> str:
    """执行劳动法赔偿计算。"""
    if monthly_salary is None or monthly_salary <= 0:
        return "错误：请提供有效的月平均工资。"

    # 日工资 ≈ 月工资 / 21.75
    daily_wage = monthly_salary / 21.75
    # 小时工资 = 日工资 / 8
    hourly_wage = daily_wage / 8

    if calc_type == "N":
        if years is None:
            return "错误：N 类型计算需要提供工作年限（years）。"
        amount = monthly_salary * years
        return (
            f"【经济补偿金（N）计算】\n"
            f"工作年限：{years} 年\n"
            f"月平均工资：{monthly_salary:.2f} 元\n"
            f"计算公式：月工资 × 工作年限\n"
            f"补偿金额：{amount:.2f} 元"
        )

    elif calc_type == "2N":
        if years is None:
            return "错误：2N 类型计算需要提供工作年限（years）。"
        amount = monthly_salary * years * 2
        return (
            f"【违法解除赔偿金（2N）计算】\n"
            f"工作年限：{years} 年\n"
            f"月平均工资：{monthly_salary:.2f} 元\n"
            f"计算公式：月工资 × 工作年限 × 2\n"
            f"赔偿金额：{amount:.2f} 元\n"
            f"法律依据：《劳动合同法》第87条"
        )

    elif calc_type == "overtime":
        weekday_pay = overtime_hours_weekday * hourly_wage * 1.5
        weekend_pay = overtime_hours_weekend * hourly_wage * 2.0
        holiday_pay = overtime_hours_holiday * hourly_wage * 3.0
        total = weekday_pay + weekend_pay + holiday_pay
        return (
            f"【加班费计算】\n"
            f"月平均工资：{monthly_salary:.2f} 元（日工资 {daily_wage:.2f} 元，小时工资 {hourly_wage:.2f} 元）\n"
            f"工作日加班：{overtime_hours_weekday:.1f} 小时 × {hourly_wage:.2f} × 1.5 = {weekday_pay:.2f} 元\n"
            f"休息日加班：{overtime_hours_weekend:.1f} 小时 × {hourly_wage:.2f} × 2.0 = {weekend_pay:.2f} 元\n"
            f"法定节假日加班：{overtime_hours_holiday:.1f} 小时 × {hourly_wage:.2f} × 3.0 = {holiday_pay:.2f} 元\n"
            f"合计加班费：{total:.2f} 元\n"
            f"法律依据：《劳动法》第44条"
        )

    elif calc_type == "annual_leave":
        if unused_leave_days is None:
            return "错误：annual_leave 类型需要提供未休年假天数（unused_leave_days）。"
        amount = unused_leave_days * daily_wage * 3
        return (
            f"【未休年假补偿计算】\n"
            f"未休年假天数：{unused_leave_days} 天\n"
            f"日平均工资：{daily_wage:.2f} 元\n"
            f"计算公式：未休天数 × 日工资 × 3\n"
            f"补偿金额：{amount:.2f} 元\n"
            f"法律依据：《职工带薪年休假条例》第5条"
        )

    else:
        return f"错误：不支持的计算类型 '{calc_type}'，支持的类型：N、2N、overtime、annual_leave"


legal_calculator = StructuredTool.from_function(
    func=_legal_calculator_impl,
    name="legal_calculator",
    description="劳动法赔偿计算器。支持四种计算类型：N（经济补偿金）、2N（违法解除赔偿金）、overtime（加班费）、annual_leave（未休年假补偿）。",
    args_schema=LegalCalculatorInput,
)


# ======================== 工具列表 ========================

def get_tools(rag_module=None, enable_web_search=True):
    """获取 Agent 工具列表。

    Args:
        rag_module: RAGModule 实例，用于 knowledge_search 工具。
        enable_web_search: 是否包含 web_search 工具。
    """
    tools = [
        _make_knowledge_search(rag_module),
        legal_calculator,
    ]
    if enable_web_search:
        tools.append(web_search)
    return tools
