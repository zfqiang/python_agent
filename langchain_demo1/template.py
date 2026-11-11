from urllib import response

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = init_chat_model(
    model="deepseek:deepseek-flash",
)

prompt_template = ChatPromptTemplate.from_template(
    "你是一个{role}专家，{description}。" \
    "用户输入：{query}" ,
    context="context",
    query="query",
)

message = prompt_template.format_prompt(role="翻译", description="将中文翻译成英文.请把下面的中文翻译成英文", query="你好，世界！")

# print(message)

# response = llm.invoke(message)
# response = llm.invoke(role="翻译", description="将中文翻译成英文.请把下面的中文翻译成英文", query="你好，世界！")
# print(response.content)

# 链式调用
chain = prompt_template | llm | StrOutputParser()

response = chain.invoke({
    "role": "翻译",
    "description": "将中文翻译成英文.请把下面的中文翻译成英文",
    "query": "你好，世界！",
})
print(response)