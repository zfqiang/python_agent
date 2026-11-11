from pyexpat import model
from urllib import response

from attr import dataclass
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.prebuilt import ToolRuntime

load_dotenv()

# 获取用户的上下文信息

@dataclass
class UserContext:
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[UserContext]) -> str:
    """
    获取当前用户所在的城市名称
    """
    user_locations = {
        "张三": "上海",
        "李四": "北京",
        "王五": "广州",
        "赵六": "深圳",
        "孙七": "杭州",
        "周八": "武汉",
        "吴九": "西安",
        "郑十": "成都",
        "王十一": "南京",
        "张十二": "苏州",
        "李十三": "无锡",
        "王十四": "常州",
    }
    return user_locations.get(runtime.context.user_id, "未知")

@tool
def get_city_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    """
    city_weather = {
        "上海": "上海天气晴转多云，最低气温 18℃",
        "北京": "北京天气阴转雷阵雨，最低气温 16℃",
        "广州": "广州天气多云转晴，最低气温 17℃",
        "深圳": "深圳天气雷阵雨转小雨，最低气温 15℃",
    }
    return city_weather.get(city, "未知城市")


llm = init_chat_model(model="deepseek:deepseek-flash")
agent = create_agent(
    model=llm,
    tools=[get_user_location, get_city_weather],
    context_schema=UserContext,
)

response = agent.invoke(
    input={"messages": [HumanMessage(content="今天的天气如何？")]},
    context=UserContext(user_id="张三")
)
# print(response["messages"][-1].content)

for message in response["messages"]:
    message.pretty_print()