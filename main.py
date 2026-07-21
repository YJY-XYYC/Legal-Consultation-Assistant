import os
import sys

# 必须在所有 huggingface 相关导入之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

from config import validate_config, create_directories
from modules.rag_module import RAGModule
from modules.agent_module import AgentModule


# 美化输出
def print_title():
    print("=" * 60)
    print("  劳动法律法规知识智能助手")
    print("=" * 60)


def print_menu():
    print("\n[表] 功能菜单:")
    print("  1. 知识问答 (RAG)")
    print("  2. 智能体任务 (Agent)")
    print("  0. 退出系统")
    print("-" * 60)


class LLMQASystem:
    def __init__(self):
        print("正在初始化系统模块...")

        # 初始化模块
        self.rag = RAGModule()
        self.agent = AgentModule()

        # 加载知识库（修复：调用正确函数名）
        try:
            count = self.rag.load_knowledge_base()
            print(f"[OK] 知识库加载完成，共 {count} 条文本片段")
        except Exception as e:
            print(f"[!]  知识库加载异常：{e}")

        print("[OK] 系统初始化完成！")

    def chat(self):
        while True:
            print_menu()
            choice = input("请输入功能编号：")

            if choice == "1":
                question = input("\n❓ 请输入你的问题：")
                answer = self.rag.ask(question)
                print("\n[i] 回答：\n", answer)

            elif choice == "2":
                task = input("\n[行] 请输入任务内容：")
                print("\n[OK] 结果：")
                result_text = ""
                for event in self.agent.run_task_stream(task):
                    if event["type"] == "answer_complete":
                        result_text = event["content"]
                    elif event["type"] == "error":
                        result_text = event["content"]
                print(result_text)

            elif choice == "0":
                print("\n👋 感谢使用，再见！")
                break

            else:
                print("[X] 无效输入，请重新选择！")


def main():
    try:
        print_title()

        # 配置检查
        errors = validate_config()
        if errors:
            print("[X] 配置错误：")
            for err in errors:
                print(f"  - {err}")
            return

        # 创建目录
        create_directories()

        # 启动系统
        system = LLMQASystem()
        system.chat()

    except Exception as e:
        print(f"\n[X] 系统启动失败: {e}")


if __name__ == "__main__":
    main()