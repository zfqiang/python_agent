from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

llm = init_chat_model(model="deepseek:deepseek-flash")

agent = create_agent(
    model=llm, 
    system_prompt="你是智能助手，根据用户提问，给出回答."
)

for chunk in agent.stream(
    {"messages": [HumanMessage(content="给我讲一个笑话")]},
    stream_mode="messages",
    version="v2",
):
    token, metadata = chunk["data"]
    if token.content:
        print(token.content, end="", flush=True)