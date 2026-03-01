import requests
import numpy as np
import os

#Local Ollama (Windows machine)
OLLAMA_HOST = "http://127.x.x.x:11434"
EMBED_MODEL = "nomic-embed-text"

#Proxmox container memory API
MEMORY_API = "http://192.168.x.x:8000"
WIKI_API = "http://192.168.x.x:8000"  


def ollama_embed(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)

def get_server_stats():
    r = requests.get(f"{MEMORY_API}/stats", timeout=10)
    r.raise_for_status()
    return r.json()

def wiki_retrieve(query_text: str, top_k: int = 6, min_score: float = 0.2):
    r = requests.post(
        f"{WIKI_API}/wiki/retrieve",
        json={"query_text": query_text, "top_k": top_k, "min_score": min_score},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("memories", [])

def net_retrieve(query_text: str, capture_id: str | None = None, top_k: int = 15, min_score: float = 0.0):
    payload = {
        "query_text": query_text,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "capture_id": capture_id or None,
    }
    r = requests.post(f"{MEMORY_API}/net/retrieve", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("results", [])

def net_import_pcap(file_path: str, timeout: float = 1800.0):
    """
    Upload .pcap/.pcapng to the server: POST /net/import_pcap (multipart form upload)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    url = f"{MEMORY_API}/net/import_pcap"

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        r = requests.post(url, files=files, timeout=timeout)

    r.raise_for_status()
    return r.json()

def net_stats(timeout: float = 30.0):
    """
    GET /net/stats
    """
    url = f"{MEMORY_API}/net/stats"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def add_memory(text: str, conversation_id: str | None = None, importance: float = 0.5, tags=None):
    if tags is None:
        tags = []
    emb = ollama_embed(text).tolist()
    r = requests.post(
        f"{MEMORY_API}/add_memory",
        json={
            "text": text,
            "embedding": emb,
            "conversation_id": conversation_id,
            "importance": importance,
            "tags": tags,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def retrieve_memories(query: str, conversation_id: str | None = None, top_k: int = 8, min_score: float = 0.25):
    q_emb = ollama_embed(query).tolist()
    r = requests.post(
        f"{MEMORY_API}/retrieve_memories",
        json={
            "query_embedding": q_emb,
            "conversation_id": conversation_id,
            "top_k": top_k,
            "min_score": min_score,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["memories"]