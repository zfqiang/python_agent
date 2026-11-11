from typing import NotRequired
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage
from typing import Callable, NotRequired
from langchain.agents.middleware import wrap_model_call,ModelRequest, ModelResponse
import os

load_dotenv()

llm = init_chat_model(model="deepseek:deepseek-flash")

vip_llm = init_chat_model(
     # 模型版本 千问大模型调用方式，其他模型有请求符
     model = "qwen3.7-plus",
     # 模型提供者，目前支持openai和huggingface
     model_provider="openai",
     # openai的api key，如果使用的是huggingface的模型，则不需要填写
     openai_api_key=os.getenv('YUNBAI_API_KEY'),
     # openai的api base，如果使用的是huggingface的模型，则不需要填写
     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class CustomStatus(AgentState):
    user_level: NotRequired[int] # 0 普通用户 1 VIP用户

@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler: Callable[[ModelRequest, ModelResponse], ModelResponse]) -> ModelResponse:
    user_level = request.state.get("user_level", 0)
    target_model = vip_llm if user_level == 1 else llm
    return handler(request.override(model=target_model))


agent = create_agent(
    model=llm, 
    state_schema=CustomStatus,
    middleware=[dynamic_model_selector],
)

response = agent.invoke({
    "messages": [HumanMessage(content="你是谁")],
    "user_level": 1
})

print(response)
