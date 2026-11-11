from json import tool

from dotenv import load_dotenv
from langchain.agents import create_agent
import os

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"今天的{city}天气是晴朗的。!"


agent = create_agent(
    model="deepseek:deepseek-flash", # 模型前缀:模型名称
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "厦门的天气如何"}]}
)
print(result["messages"][-1].content_blocks)