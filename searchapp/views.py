from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import os
import requests
from bs4 import BeautifulSoup

import numpy as np
from sentence_transformers import SentenceTransformer

from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

# Optional tokenization using tiktoken
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# ----------------------------
# App Config
# ----------------------------
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = "html_chunks"

EMBEDDING_DIM = 384
MAX_TOKENS = 500    # chunk limit


# ----------------------------
# Milvus Setup
# ----------------------------
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

def get_or_create_collection():
    """
    Create the Milvus collection only if it doesn't already exist.
    """
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="html_block", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(fields, description="Stores extracted <div> chunks")
    collection = Collection(COLLECTION_NAME, schema)

    # Basic HNSW index — keeps search fast
    index_cfg = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("embedding", index_cfg)
    collection.load()

    return collection


collection = get_or_create_collection()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# Helpers
# ----------------------------
def fetch_html(url: str) -> str:
    resp = requests.get(
        url,
        headers={"User-Agent": "html-scraper/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def extract_div_elements(html: str):
    """
    Pull out each <div> and return (plain_text, html_string)
    """
    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div")

    output = []
    for div in divs:
        text = div.get_text(" ", strip=True)
        if text:
            output.append((text, str(div)))

    return output


def chunk_by_tokens(text: str, limit: int = MAX_TOKENS):
    """
    Break a large text into smaller pieces.
    If tiktoken is not installed, fall back to word-based splitting.
    """
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        chunks = []
        for i in range(0, len(tokens), limit):
            sub = tokens[i:i + limit]
            chunks.append((enc.decode(sub), len(sub)))
        return chunks

    # fallback: simple word split
    words = text.split()
    chunks = []
    for i in range(0, len(words), limit):
        segment = words[i:i + limit]
        chunks.append((" ".join(segment), len(segment)))
    return chunks


def embed(text_list):
    emb = embedder.encode(text_list, convert_to_numpy=True)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return emb / norm


# ----------------------------
# DRF Views
# ----------------------------
class IndexView(APIView):
    """
    Extract all <div> tags from a webpage, embed them,
    and store them in Milvus.
    """
    def post(self, request):
        url = request.data.get("url")
        if not url:
            return Response({"error": "URL is required"}, status=400)

        try:
            raw_html = fetch_html(url)
            div_data = extract_div_elements(raw_html)

            if not div_data:
                return Response({"error": "No <div> blocks found"}, status=404)

            texts = [t for t, _ in div_data]
            html_blobs = [h for _, h in div_data]
            token_counts = [len(t.split()) for t in texts]

            vectors = embed(texts).tolist()

            # Insert in the same order as schema (except id)
            entities = [
                [url] * len(texts),
                texts,
                html_blobs,
                token_counts,
                vectors,
            ]

            collection.insert(entities)
            collection.flush()

            return Response(
                {"url": url, "indexed": len(texts)},
                status=status.HTTP_201_CREATED
            )

        except Exception as exc:
            return Response({"error": str(exc)}, status=400)


class SearchView(APIView):
    """
    Perform semantic search inside Milvus.
    """
    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query required"}, status=400)

        q_vec = embed([query])[0]

        collection.load()
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 200}
        }

        res = collection.search(
            data=[q_vec.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=["chunk_text", "html_block", "token_count", "url"]
        )

        out = []
        for hit in res[0]:
            out.append({
                "score": float(hit.score),
                "text": hit.entity.get("chunk_text"),
                "html": hit.entity.get("html_block"),
                "url": hit.entity.get("url"),
            })

        return Response({"results": out})
