from openai import OpenAI
import gradio as gr
import os


client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# input_text = "五一假期，爸妈和我一起去三亚海边度假"
input_text = "在北京，小明和小红在下午两点钟相约去天坛公园散步"

"""对任务进行描述"""
instruction= """
你的任务是识别用户输入的的信息
提取出对应的时间(Time)，地点(Locations)、人物(Character)
"""

"""输出"""
output = """
并已JSON格式输出
"""

"""例子"""
examples = """
五一假期，爸妈和我一起去三亚海边度假
{"时间": "五一假期", "地点": "三亚海边", "人物": "["爸妈", "我"]"}
"""

examples = """
在北京，小明和小红在下午两点钟相约去天坛公园散步
{"时间": "下午两点钟", "地点": "天坛公园", "人物": "["小明", "小红"]"}
"""

"""完整描述"""
prompt = f"""
{instruction}

{output}

{examples}

用户输入：
{input_text}
"""

# print(prompt)

def get_completion(prompt):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0, # 模型输出的随机性， 0 表示模型输出最确定的结果，1 表示模型输出最随机的结果
        # stream=False,
        # reasoning_effort="high",
        # extra_body={"thinking": {"type": "enabled"}},
    )
    return response.choices[0].message.content

print(get_completion(prompt))
# if __name__ == "__main__":
#     iface.launch(share=False)