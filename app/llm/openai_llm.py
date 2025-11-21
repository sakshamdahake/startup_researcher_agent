import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openai-gpt-oss-20b",
    temperature = 0.0,
)

# if __name__ == "__main__":
#     from langchain_core.messages import HumanMessage
    
#     llm = get_llm()
#     print("LLM object created:", llm)

#     prompt = "thank saksham for his good work"
#     print("\nSending test request to LLM...")
    
#     try:
#         result = llm.invoke([HumanMessage(content=prompt)])
#         print("\n--- LLM RAW RESPONSE ---")
#         print(result.content)
#         print("------------------------")
#     except Exception as e:
#         print("\n⚠️ LLM invocation failed!")
#         print("Error:", e)