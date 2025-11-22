import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openai-gpt-oss-120b",
    temperature = 0.0,
)