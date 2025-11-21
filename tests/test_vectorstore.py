from app.rag.vectorstore import init_index, upsert_documents, describe_index
import uuid

def test_pinecone_connection_and_upsert():
    # 1. Create index (if not exists)
    init_index()
    
    # 2. Dummy record
    test_data = [{
        "_id": f"test-{uuid.uuid4()}",
        "chunk_text": "The LangChain framework helps orchestrate LLM-powered apps.",
        "category": "llm"
    }]
    
    # 3. Upsert
    upsert_documents(test_data)

    # 4. Check index stats (optional)
    stats = describe_index()
    assert stats["total_vector_count"] > 0
