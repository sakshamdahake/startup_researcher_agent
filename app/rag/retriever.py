import os
from typing import List
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_core.documents import Document

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)


def _pinecone_hit_to_document(hit) -> Document:
    """
    Convert Pinecone hit -> LangChain Document.
    """
    return Document(
        page_content=hit["fields"].get("chunk_text", ""),
        metadata={
            "_id": hit["_id"],
            "score": hit.get("_score", None),
            **hit.get("fields", {}),
        },
    )


def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """
    Query Pinecone (integrated embedding index) and return LangChain Documents.

    This is the ONLY correct way for your index type.

    Args:
        query: text query
        k: top_k results
    """
    index = pc.Index(PINECONE_INDEX)

    response = index.search(
        namespace=PINECONE_NAMESPACE,
        query={
            "top_k": k,
            "inputs": {"text": query}
        },
    )

    hits = response.get("result", {}).get("hits", [])

    docs = [_pinecone_hit_to_document(h) for h in hits]
    return docs


class PineconeIntegratedRetriever:
    """
    A retriever class compatible with LCEL / agents / RAG chains
    """

    def __init__(self, k: int = 5):
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        return get_relevant_documents(query, k=self.k)

    def __call__(self, query: str):
        return self.invoke(query)

