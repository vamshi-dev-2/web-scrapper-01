from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os

# Try tiktoken if available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# ---------- CONFIG ----------
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = "html_chunks"
EMBED_DIM = 384
MAX_TOKENS = 500

# Connect to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Ensure Milvus collection
def ensure_collection():
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="HTML chunks")
    collection = Collection(COLLECTION_NAME, schema=schema)
    collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}})
    collection.load()
    return collection

collection = ensure_collection()
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- UTILS ----------
def fetch_html(url):
    r = requests.get(url, headers={"User-Agent": "html-search-bot/1.0"}, timeout=10)
    r.raise_for_status()
    return r.text

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def tokenize_and_chunk(text, max_tokens=MAX_TOKENS):
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
        chunks = []
        for i in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[i:i+max_tokens]
            chunk_text = enc.decode(chunk_ids)
            chunks.append((chunk_text, len(chunk_ids)))
        return chunks
    else:
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunk_words = words[i:i+max_tokens]
            chunk_text = " ".join(chunk_words)
            chunks.append((chunk_text, len(chunk_words)))
        return chunks

def embed_texts(texts):
    embs = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embs / norms

# ---------- API VIEWS ----------
class IndexView(APIView):
    def post(self, request):
        url = request.data.get("url")
        if not url:
            return Response({"error": "URL is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            html = fetch_html(url)
            text = extract_text(html)
            chunks = tokenize_and_chunk(text)
            chunk_texts = [c[0] for c in chunks]
            token_counts = [c[1] for c in chunks]
            embeddings = embed_texts(chunk_texts)

            entities = [
                [url] * len(chunk_texts),
                chunk_texts,
                token_counts,
                embeddings.tolist(),
            ]
            collection.insert(entities)
            collection.flush()

            return Response({"url": url, "chunks_indexed": len(chunk_texts)}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class SearchView(APIView):
    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query required"}, status=status.HTTP_400_BAD_REQUEST)

        # Compute query embedding
        q_emb = embed_texts([query])[0]
        collection.load()
        search_params = {"metric_type": "IP", "params": {"ef": 50}}

        # Run Milvus semantic search
        search_results = collection.search(
            data=[q_emb.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=["chunk_text", "token_count", "url"]
        )

        # Build response
        hits = []
        for hit in search_results[0]:
            snippet = hit.entity.get("chunk_text")
            token_count = int(hit.entity.get("token_count"))
            url = hit.entity.get("url")

            hits.append({
                "score": float(hit.score),       # ✅ Correct variable
                "snippet": snippet,
                "token_count": token_count,
                "url": url
            })

        return Response({"results": hits})
