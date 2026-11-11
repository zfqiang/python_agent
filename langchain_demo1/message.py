from dotenv import load_dotenv
import os
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()

llm = init_chat_model(
    model="deepseek:deepseek-flash",
)

agent = create_agent(
    model=llm,
)

messages = [
    SystemMessage(content="你是一个翻译专家，将中文翻译成英文."), # 系统设定
    HumanMessage(content="你好，世界"), # 输入消息
    AIMessage(content="hello world"), # 输出答案
    HumanMessage(content="我正在写博客"), # 当前问题
]

# response = llm.invoke(messages)
# print(response.content)

response = agent.invoke({"messages": messages})
print(response["messages"][-1].content)