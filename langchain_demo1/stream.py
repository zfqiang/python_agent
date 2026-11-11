from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import asyncio

load_dotenv()

llm = init_chat_model(model="deepseek:deepseek-flash")

# 同步调用
# for chunk in llm.stream("给我讲个冷笑话"):
#     print(chunk.content, end="", flush=True)

# 异步调用
async def steam_response():
    async for chunk in llm.astream("给我讲个冷笑话"):
        print(chunk.content, end="", flush=True)

async def main():
    # 使用asyncio.create_task 将其丢进后台事件循环中，它不会阻塞当前位置
    task = asyncio.create_task(steam_response())

    # 主进程执行任务
    print("\n[Main Task]正在执行其他任务")

    # 主进程等待大模型流水输出完毕
    await task

asyncio.run(main())