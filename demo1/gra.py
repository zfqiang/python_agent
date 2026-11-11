import gradio as gr # pip install gradio
import os
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# 定义函数，字符串反转
def reverse_string(input_string):
    return input_string[::-1]

def get_completion(content):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content},
        ],
        # stream=False,
        # reasoning_effort="high",
        # extra_body={"thinking": {"type": "enabled"}},
    )

    return response.choices[0].message.content

# type="message" 时，history 
def chat_fn(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant"}]
    messages.extend(history) #加上历史对话
    messages.append({"role": "user", "content": message}) #加上本轮对话

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        # stream=False,
        # reasoning_effort="high",
        # extra_body={"thinking": {"type": "enabled"}},
    )

    return response.choices[0].message.content

# 创建 Gradio 接口
# iface = gr.Interface(
#     fn=get_completion, # 核心函数
#     inputs="text", # 输入类型
#     outputs="text" # 输出类型
# )

# 创建 Gradio 接口
iface = gr.ChatInterface(
    fn=chat_fn, # 核心函数
    # type="messages", # 使用OpenAI风格的消息格式
    title="DeepSeek 聊天助手",
    examples=["你好", "用一句话解释什么是 Gradio"],
)

# 启动接口
if __name__ == "__main__":
    iface.launch()