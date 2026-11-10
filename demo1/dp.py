# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁"},
    ],
    # stream=False,
    # reasoning_effort="high",
    # extra_body={"thinking": {"type": "enabled"}},
)

print(response.choices[0].message.content)