from openai import OpenAI
import gradio as gr
import os


client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# 零样本COT提示
"""完整描述"""
prompt = """
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。
然后我吃了1个又去买了5个苹果。我还剩下多少苹果？
"""

# 有样本COT提示  链式思考
"""完整描述"""
prompt2 = """
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。
然后我吃了1个又去买了5个苹果。我还剩下多少苹果？
让我们逐步思考
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