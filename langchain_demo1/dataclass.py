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


@dataclass
class UserContext:
    user_id: str
    user_level: int = 2 # 1: 高级用户 : 普通用户

@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    """
    获取当前用户信息，并根据等级返回不同个的权限说明
    """
    ctx = runtime.context
    if ctx.user_level == 1:
        return f"你好{ctx.user_id}(VIP用户)，你可以执行高级操作"
    else:
        return f"你好{ctx.user_id}(普通用户)，你可以执行普通操作"

llm = init_chat_model(model="deepseek:deepseek-flash")
agent = create_agent(
    model=llm,
    tools=[get_user_info],
    context_schema=UserContext,
    system_prompt="你是一个客服助手。了解用户身份后提供相应服务。"
)

response = agent.invoke(
    input={"messages": [HumanMessage(content="请告知当前用户的权限")]},
    context=UserContext(user_id="张三", user_level=1)
)
print(response["messages"][-1].content)