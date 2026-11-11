from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi

load_dotenv()

# 初始化模型 -- 字符串模式指定模型
# llm = init_chat_model(
#     #  model="deepseek:deepseek-flash", # 模型前缀:模型名称

#      # 模型版本 千问大模型调用方式，其他模型有请求符
#      model = "qwen3.7-plus",
#      # 模型提供者，目前支持openai和huggingface
#      model_provider="openai",
#      # openai的api key，如果使用的是huggingface的模型，则不需要填写
#      openai_api_key=os.getenv('YUNBAI_API_KEY'),
#      # openai的api base，如果使用的是huggingface的模型，则不需要填写
#      openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# 初始化模型 -- 兼容Opeai API接口
llm = ChatOpenAI(
    model="qwen3.7-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv('YUNBAI_API_KEY'),
)

# 初始化模型  通过特定的模型名称指定模型
llm = ChatTongyi(
    model="qwen3.7-plus",
    api_key=os.getenv('YUNBAI_API_KEY')
)

# 调用模型
response = llm.invoke("你好，你是谁！") # 与大模型对话，返回结果

print(response)