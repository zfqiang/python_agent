from email import message

from openai import OpenAI
import gradio as gr
import os
import json
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# 初始化 deepseek 客户端
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# 初始化 qwen 客户端
qwclient = OpenAI(
    api_key=os.getenv('YUNBAI_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义获取 dp 完成度的函数
def get_dp_completion(prompt, model="deepseek-flash"):

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def get_embedding_batch(texts, model="qwen3.7-text-embedding", batch_size=16):
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = qwclient.embeddings.create(input=batch, model=model)
            all_embeddings.extend([x.embedding for x in resp.data])
        except Exception as e:
            raise RuntimeError(f"embedding failed on batch {i}-{min(i + batch_size, len(texts))}: {e}") from e
    return all_embeddings

with open('./mixed_partially_shuffled.json', 'r', encoding='utf-8-sig') as f:
    text = f.read().strip()

if not text:
    raise ValueError("JSON file is empty.")

if text.startswith('['):
    data = json.loads(text)
else:
    data = [json.loads(line) for line in text.splitlines() if line.strip()]

instructions = [entry['instruction'] for entry in data[0:100]]
outputs = [entry['output'] for entry in data[0:100]]

class MyVectorDBConnector:
    def __init__(self, embedding_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client.reset()
        self.collection = chroma_client.get_or_create_collection(name=embedding_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, instructions, outputs):
        embeddings = self.embedding_fn(instructions)
        if len(embeddings) != len(instructions) or len(instructions) != len(outputs):
            raise ValueError("Number of embeddings must match number of instructions")
        self.collection.add(
            embeddings=embeddings,
            documents=outputs,
            ids=[f"id{i}" for i in range(len(outputs))]
        )
    def search(self, query, top_n=5):
        query_embedding = self.embedding_fn([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_n
        )
        return results

vector_db = MyVectorDBConnector("demo31", get_embedding_batch)
vector_db.add_documents(instructions, outputs)

class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        search_results = self.vector_db.search(user_query, top_n=self.n_results)

        prompt = f"""
        你是一个问答机器人。
        你的任务是根据下述给定的已知信息回答用户问题。
        确保你的回复完全依据下述已知信息。不要编造答案
        如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答这个问题"

        已知信息：
        {search_results}


        用户问题：
        {user_query}

        请用中文回答用户问题。
        """

        response = self.llm_api(prompt)
        return response

bot = RAG_Bot(vector_db, get_dp_completion)

user_query = "大模型是什么东西"
response = bot.chat(user_query)
print(response)

# 用户提问后，先把问题用同一个向量模型转成向量，再去 ChromaDB 里检索出最相似的文本片段，最后把这些片段和问题一起塞进提示词，交给 DeepSeek 生成答案。