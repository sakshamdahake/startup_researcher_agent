import os
import pytest
from app.llm.openai_llm import get_llm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set → skipping live LLM test.",
)
def test_llm_live_ping(capsys):
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)

    prompt = "Reply ONLY with the word: ping"
    result = llm.invoke([HumanMessage(content=prompt)])

    # Extract text
    text = result.content.strip()

    # **Print the raw response**
    print("\n--- LLM RAW RESPONSE ---")
    print(text)
    print("------------------------\n")

    # Assert correctness
    assert "ping" in text.lower()