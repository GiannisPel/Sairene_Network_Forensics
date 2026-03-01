import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional
from embedder import embed_text

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import Query, UploadFile, File
import tempfile

NET_DIR = os.environ.get("NET_DATA_DIR", "/var/lib/memory_service/net")
NET_DB_PATH = os.path.join(NET_DIR, "net.db")
NET_FAISS_PATH = os.path.join(NET_DIR, "net.faiss")
DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "/var/lib/memory_service")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
FAISS_PATH = os.path.join(DATA_DIR, "memory.index")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Local Memory Service")

WIKI_DATA_DIR = os.environ.get("WIKI_DATA_DIR", "/var/lib/memory_service_wiki")
WIKI_DB_PATH = os.path.join(WIKI_DATA_DIR, "wiki.db")
WIKI_FAISS_PATH = os.path.join(WIKI_DATA_DIR, "wiki.faiss")
WIKI_DIM_PATH = os.path.join(WIKI_DATA_DIR, "wiki.dim")

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def _human(n: int) -> str:
    units = ["B","KB","MB","GB","TB","PB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} B"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@app.get("/stats")
def stats():
    #Personal memory store
    memory_dir = os.environ.get("MEMORY_DATA_DIR", "/var/lib/memory_service")
    wiki_dir = os.environ.get("WIKI_DATA_DIR", "/var/lib/memory_service_wiki")

    memory_bytes = _dir_size_bytes(memory_dir) if os.path.exists(memory_dir) else 0
    wiki_bytes = _dir_size_bytes(wiki_dir) if os.path.exists(wiki_dir) else 0

    return {
        "ok": True,
        "memory_store_path": memory_dir,
        "wiki_store_path": wiki_dir,
        "memory_store_size_bytes": memory_bytes,
        "wiki_store_size_bytes": wiki_bytes,
        "memory_store_size_human": _human(memory_bytes),
        "wiki_store_size_human": _human(wiki_bytes),
    }


def wiki_db():
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(WIKI_DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS wiki_memories(
        memory_id TEXT PRIMARY KEY,
        title TEXT,
        chunk INTEGER,
        text TEXT,
        created_at REAL,
        meta_json TEXT,
        faiss_row INTEGER
    )
    """)
    conn.commit()
    return conn

def wiki_get_dim() -> Optional[int]:
    if not os.path.exists(WIKI_DIM_PATH):
        return None
    with open(WIKI_DIM_PATH, "r", encoding="utf-8") as f:
        return int(f.read().strip())

def wiki_set_dim(dim: int) -> None:
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    with open(WIKI_DIM_PATH, "w", encoding="utf-8") as f:
        f.write(str(dim))

def wiki_load_or_create_index(dim: int):
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    if os.path.exists(WIKI_FAISS_PATH):
        return faiss.read_index(WIKI_FAISS_PATH)
    return faiss.IndexFlatIP(dim)

def wiki_save_index(idx) -> None:
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    faiss.write_index(idx, WIKI_FAISS_PATH)

class WikiAddTextReq(BaseModel):
    title: str
    chunk: int
    text: str
    meta: Optional[Dict[str, Any]] = None

class WikiRetrieveReq(BaseModel):
    query_text: str
    top_k: int = 8
    min_score: float = 0.2

class NetAddTextReq(BaseModel):
    text: str
    capture_id: str
    tags: List[str] = []
    importance: float = 0.7
    meta: Dict[str, Any] = {}

class NetRetrieveReq(BaseModel):
    query_text: str
    capture_id: Optional[str] = None
    top_k: int = 8
    min_score: float = 0.2

@app.get("/wiki/health")
def wiki_health():
    return {"ok": True}

@app.post("/wiki/add_text")
def wiki_add_text(req: WikiAddTextReq):
    emb = embed_text(req.text)
    dim = wiki_get_dim()
    if dim is None:
        dim = int(emb.shape[0])
        wiki_set_dim(dim)
    if int(emb.shape[0]) != dim:
        raise HTTPException(status_code=400, detail=f"Wiki dim {emb.shape[0]} != expected {dim}")

    idx = wiki_load_or_create_index(dim)
    faiss_row = int(idx.ntotal)
    idx.add(emb.reshape(1, -1))
    wiki_save_index(idx)

    now = time.time()
    memory_id = str(uuid.uuid4())
    meta = req.meta or {}
    meta.update({"title": req.title, "chunk": req.chunk, "source": "enwiki-xml"})

    conn = wiki_db()
    conn.execute(
        """INSERT INTO wiki_memories(memory_id, title, chunk, text, created_at, meta_json, faiss_row)
           VALUES(?,?,?,?,?,?,?)""",
        (memory_id, req.title, int(req.chunk), req.text, now, json.dumps(meta, ensure_ascii=False), faiss_row)
    )
    conn.commit()
    conn.close()

    return {"ok": True, "memory_id": memory_id}

@app.post("/net/import_pcap")
async def net_import_pcap(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".pcap", ".pcapng"}:
        raise HTTPException(status_code=400, detail="Only .pcap/.pcapng supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from net_pcap_ingest import ingest_pcap_file
        #Use the original filename as the capture_id
        capture_id = file.filename
        added = ingest_pcap_file(tmp_path, capture_id=file.filename)
        return {"ok": True, "capture_id": capture_id, "chunks_added": int(added)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@app.post("/wiki/retrieve")
def wiki_retrieve(req: WikiRetrieveReq):
    dim = wiki_get_dim()
    if dim is None or not os.path.exists(WIKI_FAISS_PATH):
        return {"memories": []}

    idx = faiss.read_index(WIKI_FAISS_PATH)
    q = embed_text(req.query_text)
    if int(q.shape[0]) != dim:
        raise HTTPException(status_code=400, detail=f"Query dim {q.shape[0]} != expected {dim}")

    D, I = idx.search(q.reshape(1, -1), req.top_k)

    conn = wiki_db()
    out = []
    for score, row in zip(D[0].tolist(), I[0].tolist()):
        if row < 0 or score < req.min_score:
            continue
        r = conn.execute(
            "SELECT memory_id, title, chunk, text, meta_json FROM wiki_memories WHERE faiss_row=?",
            (int(row),)
        ).fetchone()
        if not r:
            continue
        memory_id, title, chunk, text, meta_json = r
        out.append({
            "memory_id": memory_id,
            "title": title,
            "chunk": chunk,
            "text": text,
            "score": float(score),
            "meta": json.loads(meta_json) if meta_json else {}
        })
    conn.close()
    return {"memories": out}

@app.post("/net/add_text")
def net_add_text(req: NetAddTextReq):
    vec = embed_text(req.text)
    dim = int(vec.shape[0])

    idx = load_or_create_net_index(dim)
    faiss_row = int(idx.ntotal)
    idx.add(vec.reshape(1, -1))
    save_net_index(idx)

    now = time.time()
    mid = str(uuid.uuid4())
    meta = {
        "capture_id": req.capture_id,
        "tags": req.tags,
        "importance": float(req.importance),
        "meta": req.meta,
    }

    conn = net_db()
    conn.execute(
        "INSERT INTO net_memories(id, capture_id, text, created_at, meta_json, faiss_row) VALUES(?,?,?,?,?,?)",
        (mid, req.capture_id, req.text, now, json.dumps(meta, ensure_ascii=False), faiss_row),
    )
    conn.commit()
    conn.close()

    return {"ok": True, "id": mid}

@app.post("/net/retrieve")
def net_retrieve(req: NetRetrieveReq):
    q = embed_text(req.query_text)
    dim = int(q.shape[0])

    idx = load_or_create_net_index(dim)
    if idx.ntotal == 0:
        return {"ok": True, "results": []}

    #What user wants back
    want = max(1, min(int(req.top_k), 50))

    #Oversample to survive filtering by capture_id
    oversample = min(max(want * 5, 25), 200)

    D, I = idx.search(q.reshape(1, -1), oversample)

    conn = net_db()
    results = []

    for score, faiss_row in zip(D[0].tolist(), I[0].tolist()):
        if faiss_row < 0:
            continue
        if float(score) < float(req.min_score):
            continue

        cur = conn.execute(
            "SELECT id, capture_id, text, meta_json FROM net_memories WHERE faiss_row=?",
            (int(faiss_row),),
        )
        row = cur.fetchone()
        if not row:
            continue

        mid, cap, text, meta_json = row

        #Apply capture filter after the row
        if req.capture_id and cap != req.capture_id:
            continue

        results.append({
            "id": mid,
            "capture_id": cap,
            "text": text,
            "score": float(score),
            "meta": json.loads(meta_json) if meta_json else {},
        })

        #Stop once I have what user requested
        if len(results) >= want:
            break

    conn.close()
    return {"ok": True, "results": results}


@app.post("/net/reset")
def net_reset():
    ensure_dir(NET_DIR)
    if os.path.exists(NET_DB_PATH):
        os.remove(NET_DB_PATH)
    if os.path.exists(NET_FAISS_PATH):
        os.remove(NET_FAISS_PATH)
    return {"ok": True}

@app.get("/net/captures")
def net_captures():
    conn = net_db()
    cur = conn.execute(
        "SELECT capture_id, COUNT(*) FROM net_memories GROUP BY capture_id ORDER BY COUNT(*) DESC"
    )
    rows = [{"capture_id": r[0], "count": int(r[1])} for r in cur.fetchall()]
    conn.close()
    return {"ok": True, "captures": rows}


#Top 10 IPS from capture
@app.get("/net/viz/top-ips")
def net_viz_top_ips(capture_id: str, limit: int = 10):
    conn = net_db()
    cur = conn.execute("""
    SELECT 
        json_extract(meta_json, '$.layers.network.src_ip') as ip,
        COUNT(*) as count
    FROM net_memories
    WHERE capture_id = ?
    GROUP BY ip
    ORDER BY count DESC
    LIMIT ?
    """, (capture_id, limit)
    )

    rows = cur.fetchall()
    return [{"ip": r[0], "count": r[1]} for r in rows]

#Sankey Diagram, shows SRC->Port->DST
@app.get("/net/viz/flow")
def net_viz_flow(capture_id: str):
    conn = net_db()
    cur = conn.execute("""
    SELECT
        json_extract(meta_json, '$.layers.network.src_ip') as src,
        json_extract(meta_json, '$.layers.transport.dst_port') as port,
        json_extract(meta_json, '$.layers.network.dst_ip') as dst,
        COUNT(*) as count
    FROM net_memories
    WHERE capture_id = ?
    GROUP BY src, port, dst
    ORDER BY count DESC
    """, (capture_id,)
    )

    rows = cur.fetchall()
    return [
        {
            "src": r[0],
            "port": r[1],
            "dst": r[2],
            "count": r[3]
        }
        for r in rows
    ]

@app.get("/net/stats")
def net_stats():
    conn = net_db()
    cur = conn.execute("SELECT COUNT(*) FROM net_memories")
    count = int(cur.fetchone()[0])
    conn.close()
    return {"ok": True, "count": count}

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def net_db():
    ensure_dir(NET_DIR)
    conn = sqlite3.connect(NET_DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS net_memories (
        id TEXT PRIMARY KEY,
        capture_id TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at REAL NOT NULL,
        meta_json TEXT NOT NULL,
        faiss_row INTEGER NOT NULL
    );
    """)
    conn.commit()
    return conn

def load_or_create_net_index(dim: int):
    ensure_dir(NET_DIR)
    if os.path.exists(NET_FAISS_PATH):
        return faiss.read_index(NET_FAISS_PATH)
    idx = faiss.IndexFlatIP(dim) #Cosine similarity if vectors are normalized
    faiss.write_index(idx, NET_FAISS_PATH)
    return idx

def save_net_index(idx):
    ensure_dir(NET_DIR)
    faiss.write_index(idx, NET_FAISS_PATH)

def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_used_at REAL NOT NULL,
        importance REAL NOT NULL,
        meta_json TEXT NOT NULL,
        faiss_row INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        conversation_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """)

    conn.commit()
    conn.close()

def get_dim() -> Optional[int]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key='dim'")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def set_dim(dim: int) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO config(key,value) VALUES('dim', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(dim),)
    )
    conn.commit()
    conn.close()

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return (v / n).astype(np.float32)

def load_or_create_index(dim: int) -> faiss.Index:
    if os.path.exists(FAISS_PATH):
        idx = faiss.read_index(FAISS_PATH)
        if idx.d != dim:
            raise RuntimeError(f"FAISS dim mismatch: index={idx.d}, expected={dim}")
        return idx
    idx = faiss.IndexFlatIP(dim)  #Cosine if normalized
    faiss.write_index(idx, FAISS_PATH)
    return idx

def save_index(idx: faiss.Index) -> None:
    faiss.write_index(idx, FAISS_PATH)

init_db()

class AddMessageReq(BaseModel):
    conversation_id: str
    role: str
    content: str

class AddMemoryReq(BaseModel):
    text: str
    embedding: List[float]              #Send embedding from Windows (Ollama)
    conversation_id: Optional[str] = None
    importance: float = 0.5
    tags: List[str] = []

class RetrieveReq(BaseModel):
    query_embedding: List[float]
    top_k: int = 8
    min_score: float = 0.25
    conversation_id: Optional[str] = None  #If I want scoped memory later

@app.get("/search_memories")
def search_memories(query: str, conversation_id: str | None = None, limit: int = 20):
    conn = db()
    cur = conn.cursor()

    q = f"%{query}%"
    if conversation_id:
        cur.execute(
            """
            SELECT memory_id, text, importance, meta_json, created_at
            FROM memories
            WHERE text LIKE ? AND meta_json LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (q, f'%"{conversation_id}"%', limit),
        )
    else:
        cur.execute(
            """
            SELECT memory_id, text, importance, meta_json, created_at
            FROM memories
            WHERE text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (q, limit),
        )

    rows = cur.fetchall()
    conn.close()

    out = []
    for memory_id, text, importance, meta_json, created_at in rows:
        out.append({
            "memory_id": memory_id,
            "text": text,
            "importance": float(importance),
            "meta": json.loads(meta_json),
            "created_at": float(created_at),
        })
    return {"memories": out}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/add_message")
def add_message(req: AddMessageReq):
    conn = db()
    conn.execute(
        "INSERT INTO messages(id, conversation_id, role, content, created_at) VALUES(?,?,?,?,?)",
        (str(uuid.uuid4()), req.conversation_id, req.role, req.content, time.time())
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/add_memory")
def add_memory(req: AddMemoryReq):
    emb = np.array(req.embedding, dtype=np.float32)

    dim = get_dim()
    if dim is None:
        dim = int(emb.shape[0])
        set_dim(dim)
    if emb.shape[0] != dim:
        raise HTTPException(status_code=400, detail=f"Embedding dim {emb.shape[0]} != expected {dim}")

    idx = load_or_create_index(dim)

    #Normalize for cosine similarity (IndexFlatIP)
    emb = normalize(emb)

    #Store normalized embedding in DB (needed for safe rebuilds later)
    embedding_bytes = emb.tobytes()

    faiss_row = int(idx.ntotal)
    idx.add(emb.reshape(1, -1))
    save_index(idx)

    meta = {
        "conversation_id": req.conversation_id,
        "tags": req.tags,
        "source": "client",
    }

    now = time.time()
    memory_id = str(uuid.uuid4())
    conn = db()
    conn.execute(
        """INSERT INTO memories(memory_id, text, created_at, last_used_at, importance, meta_json, faiss_row, embedding)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            memory_id,
            req.text,
            now,
            now,
            float(req.importance),
            json.dumps(meta, ensure_ascii=False),
            faiss_row,
            embedding_bytes,
        ),
    )
    conn.commit()
    conn.close()

    return {"ok": True, "memory_id": memory_id}


@app.delete("/delete_memory")
def delete_memory(memory_id: str = Query(...)):
    """Delete memory by its memory_id"""
    conn = db()
    try:
        cur = conn.cursor()
    	#Delete from SQLite
        cur.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        rowcount = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if rowcount == 0:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"ok": True, "deleted": memory_id}

def rebuild_faiss_index():
    """Rebuild FAISS from SQLite memory table"""
    dim = get_dim()
    if dim is None:
        return

    #Create empty index
    idx = faiss.IndexFlatIP(dim)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT memory_id, text, meta_json FROM memories ORDER BY created_at ASC")
    rows = cur.fetchball()

    for faiss_row, row in enumerate(rows):
        memory_id, text, meta_json = row
        #Do not re-embed, embeddings are not stored in SQLite yet.
        #Upgrade later
        emb_bytes = row [0]
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        emb = normalize(emb)
        idx.add(emb.reshape(1,-1))
    
    conn.close()

    faiss.write_index(idx, FAISS_PATH)

@app.post("/retrieve_memories")
def retrieve_memories(req: RetrieveReq):
    q = np.array(req.query_embedding, dtype=np.float32)
    dim = get_dim()
    if dim is None:
        return {"memories": []}
    if q.shape[0] != dim:
        raise HTTPException(status_code=400, detail=f"Query dim {q.shape[0]} != expected {dim}")

    if not os.path.exists(FAISS_PATH):
        return {"memories": []}

    idx = faiss.read_index(FAISS_PATH)
    q = normalize(q)

    k = min(req.top_k, int(idx.ntotal))
    if k <= 0:
        return {"memories": []}

    scores, rows = idx.search(q.reshape(1, -1), k)
    scores = scores.flatten().tolist()
    rows = rows.flatten().tolist()

    conn = db()
    cur = conn.cursor()

    out = []
    now = time.time()
    for faiss_row, score in zip(rows, scores):
        if faiss_row < 0 or score < req.min_score:
            continue
        cur.execute(
            "SELECT memory_id, text, created_at, last_used_at, importance, meta_json FROM memories WHERE faiss_row=?",
            (int(faiss_row),)
        )
        r = cur.fetchone()
        if not r:
            continue

        memory_id, text, created_at, last_used_at, importance, meta_json = r
        meta = json.loads(meta_json)
        meta["score"] = float(score)

        #scope by conversation_id (simple check)
        if req.conversation_id and meta.get("conversation_id") not in (None, req.conversation_id):
            continue

        out.append({
            "memory_id": memory_id,
            "text": text,
            "importance": float(importance),
            "meta": meta,
            "created_at": float(created_at),
        })

        cur.execute("UPDATE memories SET last_used_at=? WHERE memory_id=?", (now, memory_id))

    conn.commit()
    conn.close()
    return {"memories": out}