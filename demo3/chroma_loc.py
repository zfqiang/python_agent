from openai import OpenAI
import gradio as gr
import os
import json
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('YUNBAI_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_embedding(texts, model="qwen3.7-text-embedding"):
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

def get_embedding_batch(texts, model="qwen3.7-text-embedding", batch_size=16):
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
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

instructions = [entry['instruction'] for entry in data[0:1000]]
outputs = [entry['output'] for entry in data[0:1000]]

class MyVectorDBConnector:
    def __init__(self, embedding_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client.reset()
        self.collection = chroma_client.get_or_create_collection(name=embedding_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, instructions, outputs):
        embeddings = self.embedding_fn(instructions)
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

vector_db = MyVectorDBConnector("demo3", get_embedding_batch)
vector_db.add_documents(instructions, outputs)


user_query = "大模型是什么东西"
results = vector_db.search(user_query)
for para in results['documents'][0]:
    print(para + '\n')