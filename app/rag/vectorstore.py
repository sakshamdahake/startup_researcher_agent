import os
import logging
from typing import (
    List,
    Dict,
    Optional
)

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY must be set in environment or .env file")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pc = Pinecone(api_key=PINECONE_API_KEY)

def init_index():
    """
    Create Pinecone index with integrated embedding if it doesn't exist.
    """
    if not pc.has_index(PINECONE_INDEX):
        pc.create_index_for_model(
            name = PINECONE_INDEX,
            cloud = PINECONE_CLOUD,
            region = PINECONE_REGION,
            embed={
                "model": PINECONE_EMBED_MODEL,
                "field_map": {"text": "chunk_text"}
            }
        )
        logger.info(f"Index '{PINECONE_INDEX}' created.")
    else:
        logger.info(f"Index '{PINECONE_INDEX}' already exists.")

def upsert_documents(records: List[Dict]):
    """
    Upsert a list of chunk records into Pinecone index.

    Each record should be a dictionary with at least:
    - "_id": unique identifier
    - "chunk_text": the actual text
    """
    index = pc.Index(PINECONE_INDEX)
    index.upsert_records(namespace=PINECONE_NAMESPACE, records=records)
    logger.info(f"Upserted {len(records)} records into namespace '{PINECONE_NAMESPACE}'.")

def search_similar_documents(query: str, top_k: int=5) -> List[Dict]:
    """
    Perform a semantic search in the index using a plain-text query.

    Args:
        query (str): The search string
        top_k (int): Number of results to return

    Returns:
        List of matching document records
    """
    index = pc.Index(PINECONE_INDEX)
    response = index.search(
        namespace = PINECONE_NAMESPACE,
        query = {
            "top_k": top_k,
            "inputs": {
                "text": query
            }
        }
    )
    return response.get("result", {}).get("hits", [])

def describe_index():
    """
    Print and return stats of the Pinecone index.
    """
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")
    return stats