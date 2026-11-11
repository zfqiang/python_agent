from pyexpat import model

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

#定义工具
@tool
def check_stock(product: str) -> str:
    """检查商品库存"""
    stock_data = {
        "apple": 10,
        "banana": 5,
        "orange": 20
    }
    stock = stock_data.get(product, 0)
    if stock > 0:
        return f"{product}有{stock}件"
    else:
        return f"{product}没有库存"

#创建代理
llm = init_chat_model(
    model="deepseek:deepseek-flash",
)

#创建angent并注册工具
agent = create_agent(
    model=llm,
    tools=[check_stock],
)

response = agent.invoke({
    "messages": [
        HumanMessage(content="apple有多少件")
    ],
})
# print(response["messages"][-1].content)

for message in response["messages"]:
    message.pretty_print()

