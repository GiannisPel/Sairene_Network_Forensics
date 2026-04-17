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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi import Query, UploadFile, File
import tempfile


#Paths all configurable via environment variables

NET_DIR        = os.environ.get("NET_DATA_DIR",    "/var/lib/memory_service/net")
NET_DB_PATH    = os.path.join(NET_DIR,  "net.db")
NET_FAISS_PATH = os.path.join(NET_DIR,  "net.faiss")

DATA_DIR       = os.environ.get("MEMORY_DATA_DIR", "/var/lib/memory_service")
DB_PATH        = os.path.join(DATA_DIR, "memory.db")
FAISS_PATH     = os.path.join(DATA_DIR, "memory.index")

WIKI_DATA_DIR  = os.environ.get("WIKI_DATA_DIR",   "/var/lib/memory_service_wiki")
WIKI_DB_PATH   = os.path.join(WIKI_DATA_DIR, "wiki.db")
WIKI_FAISS_PATH= os.path.join(WIKI_DATA_DIR, "wiki.faiss")
WIKI_DIM_PATH  = os.path.join(WIKI_DATA_DIR, "wiki.dim")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Local Memory Service")


#In memory FAISS singletons
#FIXED: previously every endpoint called faiss.read_index() on every request, deserializing the entire index from disk each time. Now loaded once.
_mem_index:  Optional[faiss.Index] = None
_net_index:  Optional[faiss.Index] = None
_wiki_index: Optional[faiss.Index] = None

#Dirty flags when index written to disk only when marked dirty,
#not on every single add_memory call
_mem_index_dirty:  bool = False
_net_index_dirty:  bool = False
_wiki_index_dirty: bool = False


def get_mem_index(dim: int) -> faiss.Index:
    global _mem_index
    if _mem_index is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(FAISS_PATH):
            _mem_index = faiss.read_index(FAISS_PATH)
            if _mem_index.d != dim:
                raise RuntimeError(
                    f"FAISS dim mismatch: index={_mem_index.d}, expected={dim}"
                )
        else:
            _mem_index = faiss.IndexFlatIP(dim)
    return _mem_index

def flush_mem_index() -> None:
    global _mem_index_dirty
    if _mem_index is not None and _mem_index_dirty:
        faiss.write_index(_mem_index, FAISS_PATH)
        _mem_index_dirty = False

def get_net_index(dim: int) -> faiss.Index:
    global _net_index
    if _net_index is None:
        os.makedirs(NET_DIR, exist_ok=True)
        if os.path.exists(NET_FAISS_PATH):
            _net_index = faiss.read_index(NET_FAISS_PATH)
        else:
            _net_index = faiss.IndexFlatIP(dim)
    return _net_index

def flush_net_index() -> None:
    global _net_index_dirty
    if _net_index is not None and _net_index_dirty:
        faiss.write_index(_net_index, NET_FAISS_PATH)
        _net_index_dirty = False

def get_wiki_index(dim: int) -> faiss.Index:
    global _wiki_index
    if _wiki_index is None:
        os.makedirs(WIKI_DATA_DIR, exist_ok=True)
        if os.path.exists(WIKI_FAISS_PATH):
            _wiki_index = faiss.read_index(WIKI_FAISS_PATH)
        else:
            _wiki_index = faiss.IndexFlatIP(dim)
    return _wiki_index

def flush_wiki_index() -> None:
    global _wiki_index_dirty
    if _wiki_index is not None and _wiki_index_dirty:
        faiss.write_index(_wiki_index, WIKI_FAISS_PATH)
        _wiki_index_dirty = False

@app.on_event("shutdown")
def on_shutdown():
    """Flush all dirty indexes to disk on clean shutdown."""
    flush_mem_index()
    flush_net_index()
    flush_wiki_index()


#Helpers

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
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} B"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return (v / n).astype(np.float32)



#SQLite connection helpers

#FIXED: net_db() was missing WAL and synchronous pragmas that db() had
#So I added cache_size and temp_store for performance on limited RAM
#and added missing faiss_row indexes on all three tables

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-32000;")   #32 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY;")

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _apply_pragmas(conn)
    return conn

def net_db() -> sqlite3.Connection:
    ensure_dir(NET_DIR)
    conn = sqlite3.connect(NET_DB_PATH, check_same_thread=False)
    _apply_pragmas(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS net_memories (
            id         TEXT PRIMARY KEY,
            capture_id TEXT NOT NULL,
            text       TEXT NOT NULL,
            created_at REAL NOT NULL,
            meta_json  TEXT NOT NULL,
            faiss_row  INTEGER NOT NULL
        );
    """)
    #FIXED:missing indexes when retrieve was doing a full table scan per result
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_net_faiss_row "
        "ON net_memories(faiss_row);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_net_capture_id "
        "ON net_memories(capture_id);"
    )
    conn.commit()
    return conn

def wiki_db() -> sqlite3.Connection:
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(WIKI_DB_PATH, check_same_thread=False)
    _apply_pragmas(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wiki_memories (
            memory_id  TEXT PRIMARY KEY,
            title      TEXT,
            chunk      INTEGER,
            text       TEXT,
            created_at REAL,
            meta_json  TEXT,
            faiss_row  INTEGER
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_faiss_row "
        "ON wiki_memories(faiss_row);"
    )
    conn.commit()
    return conn

def init_db() -> None:
    conn = db()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      REAL NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            memory_id    TEXT PRIMARY KEY,
            text         TEXT NOT NULL,
            created_at   REAL NOT NULL,
            last_used_at REAL NOT NULL,
            importance   REAL NOT NULL,
            meta_json    TEXT NOT NULL,
            faiss_row    INTEGER NOT NULL,
            embedding    BLOB
        );
    """)
    #FIXED: missing index where retrieve was doing a full table scan per result
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_mem_faiss_row ON memories(faiss_row);"
    )
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            summary         TEXT NOT NULL,
            updated_at      REAL NOT NULL
        );
    """)

    conn.commit()
    conn.close()



#Dimension helpers

def get_dim() -> Optional[int]:
    conn = db()
    cur  = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key='dim'")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def set_dim(dim: int) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO config(key,value) VALUES('dim', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(dim),)
    )
    conn.commit()
    conn.close()

def wiki_get_dim() -> Optional[int]:
    if not os.path.exists(WIKI_DIM_PATH):
        return None
    with open(WIKI_DIM_PATH, "r", encoding="utf-8") as f:
        return int(f.read().strip())

def wiki_set_dim(dim: int) -> None:
    os.makedirs(WIKI_DATA_DIR, exist_ok=True)
    with open(WIKI_DIM_PATH, "w", encoding="utf-8") as f:
        f.write(str(dim))



#Pydantic models

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

class AddMessageReq(BaseModel):
    conversation_id: str
    role: str
    content: str

class AddMemoryReq(BaseModel):
    text: str
    embedding: List[float]
    conversation_id: Optional[str] = None
    importance: float = 0.5
    tags: List[str] = []

class RetrieveReq(BaseModel):
    query_embedding: List[float]
    top_k: int = 8
    min_score: float = 0.25
    conversation_id: Optional[str] = None



#Stats & health

@app.get("/stats")
def stats():
    memory_dir = os.environ.get("MEMORY_DATA_DIR", "/var/lib/memory_service")
    wiki_dir   = os.environ.get("WIKI_DATA_DIR",   "/var/lib/memory_service_wiki")

    memory_bytes = _dir_size_bytes(memory_dir) if os.path.exists(memory_dir) else 0
    wiki_bytes   = _dir_size_bytes(wiki_dir)   if os.path.exists(wiki_dir)   else 0

    return {
        "ok":                       True,
        "memory_store_path":        memory_dir,
        "wiki_store_path":          wiki_dir,
        "memory_store_size_bytes":  memory_bytes,
        "wiki_store_size_bytes":    wiki_bytes,
        "memory_store_size_human":  _human(memory_bytes),
        "wiki_store_size_human":    _human(wiki_bytes),
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/wiki/health")
def wiki_health():
    return {"ok": True}


#Wiki endpoints
@app.post("/wiki/add_text")
def wiki_add_text(req: WikiAddTextReq):
    global _wiki_index_dirty

    emb = embed_text(req.text)
    dim = wiki_get_dim()
    if dim is None:
        dim = int(emb.shape[0])
        wiki_set_dim(dim)
    if int(emb.shape[0]) != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Wiki dim {emb.shape[0]} != expected {dim}"
        )

    idx       = get_wiki_index(dim)
    faiss_row = int(idx.ntotal)
    idx.add(emb.reshape(1, -1))
    _wiki_index_dirty = True
    flush_wiki_index()  #wiki ingest is offline batch work then flush immediately

    now       = time.time()
    memory_id = str(uuid.uuid4())
    meta      = req.meta or {}
    meta.update({"title": req.title, "chunk": req.chunk, "source": "enwiki-xml"})

    conn = wiki_db()
    conn.execute(
        """INSERT INTO wiki_memories
           (memory_id, title, chunk, text, created_at, meta_json, faiss_row)
           VALUES (?,?,?,?,?,?,?)""",
        (memory_id, req.title, int(req.chunk), req.text, now,
         json.dumps(meta, ensure_ascii=False), faiss_row)
    )
    conn.commit()
    conn.close()

    return {"ok": True, "memory_id": memory_id}

@app.post("/wiki/retrieve")
def wiki_retrieve(req: WikiRetrieveReq):
    dim = wiki_get_dim()
    if dim is None or not os.path.exists(WIKI_FAISS_PATH):
        return {"memories": []}

    idx = get_wiki_index(dim)
    q   = embed_text(req.query_text)
    if int(q.shape[0]) != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Query dim {q.shape[0]} != expected {dim}"
        )

    D, I = idx.search(q.reshape(1, -1), req.top_k)

    valid = [
        (float(score), int(row))
        for score, row in zip(D[0].tolist(), I[0].tolist())
        if row >= 0 and float(score) >= req.min_score
    ]
    if not valid:
        return {"memories": []}

    #FIXED: single batch query instead of one SELECT per result
    placeholders = ",".join("?" * len(valid))
    row_ids   = [r for _, r in valid]
    score_map = {r: s for s, r in valid}

    conn    = wiki_db()
    db_rows = conn.execute(
        f"SELECT memory_id, title, chunk, text, meta_json, faiss_row "
        f"FROM wiki_memories WHERE faiss_row IN ({placeholders})",
        row_ids
    ).fetchall()
    conn.close()

    out = [
        {
            "memory_id": memory_id,
            "title":     title,
            "chunk":     chunk,
            "text":      text,
            "score":     score_map.get(faiss_row, 0.0),
            "meta":      json.loads(meta_json) if meta_json else {},
        }
        for memory_id, title, chunk, text, meta_json, faiss_row in db_rows
    ]
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"memories": out}



#Net endpoints

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
        capture_id = file.filename
        added      = ingest_pcap_file(tmp_path, capture_id=file.filename)
        return {"ok": True, "capture_id": capture_id, "chunks_added": int(added)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

#Progress bar when importing
#Where the synchronous CPUheavy ingest runs in a background thread 
#It pushes events into a queue. The async generator drains the queue and yields lines to the HTTP response 
#This lets uvicorn flush each line to the client immediately instead of buffering everything until the ingest finishes
@app.post("/net/import_pcap_stream")
async def net_import_pcap_stream(file: UploadFile = File(...)):

    import asyncio
    import queue
    import threading

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".pcap", ".pcapng"}:
        raise HTTPException(status_code=400, detail="Only .pcap/.pcapng supported")

    contents = await file.read()
    filename = file.filename

    #Write temp file before starting thread
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    #Placed in queue by the thread when it is done (sentinel)
    _DONE = object()

    event_queue: queue.Queue = queue.Queue()

    def run_ingest():
        """Runs in a background thread. Pushes string lines into event_queue."""
        try:
            from net_pcap_ingest import ingest_pcap_file_stream

            for event in ingest_pcap_file_stream(tmp_path, capture_id=filename):
                if len(event) == 2:
                    current, total = event
                    if current == 0:
                        event_queue.put(f"TOTAL {total}\n")
                    else:
                        event_queue.put(f"PROGRESS {current} {total}\n")
                elif len(event) == 3:
                    added, total, cap_id = event
                    event_queue.put(f"DONE {added} {cap_id}\n")

        except Exception as e:
            event_queue.put(f"ERROR {e}\n")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            event_queue.put(_DONE)

    #Start the ingest thread
    thread = threading.Thread(target=run_ingest, daemon=True)
    thread.start()

    async def generate():
        """
        Async generator that drains event_queue.
        Yields each line immediately so the HTTP response flushes to the
        client in real time rather than buffering until ingest completes.
        """
        loop = asyncio.get_event_loop()
        while True:
            #Run the blocking queue.get in a thread pool so it doesnt block the event loop while waiting for the next batch
            line = await loop.run_in_executor(None, event_queue.get)
            if line is _DONE:
                break
            yield line

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/net/add_text")
def net_add_text(req: NetAddTextReq):
    global _net_index_dirty

    vec = embed_text(req.text)
    dim = int(vec.shape[0])

    idx       = get_net_index(dim)
    faiss_row = int(idx.ntotal)
    idx.add(vec.reshape(1, -1))
    _net_index_dirty = True
    flush_net_index()

    now = time.time()
    mid = str(uuid.uuid4())
    meta = {
        "capture_id": req.capture_id,
        "tags":       req.tags,
        "importance": float(req.importance),
        "meta":       req.meta,
    }

    conn = net_db()
    conn.execute(
        "INSERT INTO net_memories "
        "(id, capture_id, text, created_at, meta_json, faiss_row) "
        "VALUES (?,?,?,?,?,?)",
        (mid, req.capture_id, req.text, now,
         json.dumps(meta, ensure_ascii=False), faiss_row),
    )
    conn.commit()
    conn.close()

    return {"ok": True, "id": mid}

@app.post("/net/retrieve")
def net_retrieve(req: NetRetrieveReq):
    q   = embed_text(req.query_text)
    dim = int(q.shape[0])

    idx = get_net_index(dim)
    if idx.ntotal == 0:
        return {"ok": True, "results": []}

    want       = max(1, min(int(req.top_k), 50))
    oversample = min(max(want * 5, 25), 200)

    D, I = idx.search(q.reshape(1, -1), oversample)

    valid = [
        (float(score), int(row))
        for score, row in zip(D[0].tolist(), I[0].tolist())
        if row >= 0 and float(score) >= float(req.min_score)
    ]
    if not valid:
        return {"ok": True, "results": []}

    #FIXED:single batch query instead of one SELECT per result
    placeholders = ",".join("?" * len(valid))
    row_ids   = [r for _, r in valid]
    score_map = {r: s for s, r in valid}

    conn    = net_db()
    db_rows = conn.execute(
        f"SELECT id, capture_id, text, meta_json, faiss_row "
        f"FROM net_memories WHERE faiss_row IN ({placeholders})",
        row_ids
    ).fetchall()
    conn.close()

    results = []
    for mid, cap, text, meta_json, faiss_row in db_rows:
        if req.capture_id and cap != req.capture_id:
            continue
        results.append({
            "id":         mid,
            "capture_id": cap,
            "text":       text,
            "score":      score_map.get(faiss_row, 0.0),
            "meta":       json.loads(meta_json) if meta_json else {},
        })
        if len(results) >= want:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"ok": True, "results": results}

@app.post("/net/reset")
def net_reset():
    global _net_index
    ensure_dir(NET_DIR)
    if os.path.exists(NET_DB_PATH):
        os.remove(NET_DB_PATH)
    if os.path.exists(NET_FAISS_PATH):
        os.remove(NET_FAISS_PATH)
    _net_index = None  #invalidate singleton so its rebuilt on next request
    return {"ok": True}

@app.get("/net/captures")
def net_captures():
    conn = net_db()
    cur  = conn.execute(
        "SELECT capture_id, COUNT(*) FROM net_memories "
        "GROUP BY capture_id ORDER BY COUNT(*) DESC"
    )
    rows = [{"capture_id": r[0], "count": int(r[1])} for r in cur.fetchall()]
    conn.close()
    return {"ok": True, "captures": rows}

@app.delete("/net/delete_capture")
def net_delete_capture(capture_id: str = Query(...)):
    """
    Delete all packets belonging to a specific capture_id from SQLite.

    Note: the FAISS index is NOT rebuilt automatically — deleted vectors
    leave gaps but do not cause incorrect results, they just waste a small
    amount of memory. Call /rebuild_index after bulk deletions if needed.
    """
    conn = net_db()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM net_memories WHERE capture_id = ?", (capture_id,))
    count = cur.fetchone()[0]

    if count == 0:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail=f"Capture '{capture_id}' not found."
        )

    cur.execute("DELETE FROM net_memories WHERE capture_id = ?", (capture_id,))
    conn.commit()
    conn.close()

    return {"ok": True, "capture_id": capture_id, "packets_deleted": int(count)}

@app.get("/net/viz/top-ips")
def net_viz_top_ips(capture_id: str, limit: int = 10):
    conn = net_db()
    cur  = conn.execute("""
        SELECT
            json_extract(meta_json, '$.layers.network.src_ip') AS ip,
            COUNT(*) AS count
        FROM net_memories
        WHERE capture_id = ?
        GROUP BY ip
        ORDER BY count DESC
        LIMIT ?
    """, (capture_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"ip": r[0], "count": r[1]} for r in rows]

@app.get("/net/viz/flow")
def net_viz_flow(capture_id: str):
    conn = net_db()
    cur  = conn.execute("""
        SELECT
            json_extract(meta_json, '$.layers.network.src_ip')    AS src,
            json_extract(meta_json, '$.layers.transport.dst_port') AS port,
            json_extract(meta_json, '$.layers.network.dst_ip')    AS dst,
            COUNT(*) AS count
        FROM net_memories
        WHERE capture_id = ?
        GROUP BY src, port, dst
        ORDER BY count DESC
    """, (capture_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"src": r[0], "port": r[1], "dst": r[2], "count": r[3]} for r in rows]


#Returns all anomalous records for a capture. 
#WhereFlow summary records (flow_record_type=flow_summary) are returned first
#as they represent completed behavioral analysis and are more reliable
#for detecting slow attacks like exfiltration and beaconing.
@app.get("/net/anomalies")
def net_anomalies(capture_id: str):
    conn = net_db()
    rows = conn.execute(
        "SELECT meta_json FROM net_memories WHERE capture_id=?",
        (capture_id,)
    ).fetchall()
    conn.close()

    flow_anomalies   = []
    packet_anomalies = []

    for r in rows:
        meta = json.loads(r[0])
        if not meta.get("ml", {}).get("anomaly"):
            continue
        if meta.get("flow_record_type") == "flow_summary":
            flow_anomalies.append(meta)
        else:
            packet_anomalies.append(meta)

    #Flow summaries first cause they represent complete behavioral evidence
    return flow_anomalies + packet_anomalies

@app.get("/net/stats")
def net_stats():
    conn  = net_db()
    count = int(conn.execute("SELECT COUNT(*) FROM net_memories").fetchone()[0])
    conn.close()
    return {"ok": True, "count": count}

@app.get("/search_memories")
def search_memories(
    query: str,
    conversation_id: Optional[str] = None,
    limit: int = 20
):
    conn = db()
    cur  = conn.cursor()
    q    = f"%{query}%"

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

    return {
        "memories": [
            {
                "memory_id":  memory_id,
                "text":       text,
                "importance": float(importance),
                "meta":       json.loads(meta_json),
                "created_at": float(created_at),
            }
            for memory_id, text, importance, meta_json, created_at in rows
        ]
    }

@app.post("/add_message")
def add_message(req: AddMessageReq):
    conn = db()
    conn.execute(
        "INSERT INTO messages(id, conversation_id, role, content, created_at) "
        "VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), req.conversation_id, req.role, req.content, time.time())
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/add_memory")
def add_memory(req: AddMemoryReq):
    global _mem_index_dirty

    emb = np.array(req.embedding, dtype=np.float32)

    dim = get_dim()
    if dim is None:
        dim = int(emb.shape[0])
        set_dim(dim)
    if emb.shape[0] != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dim {emb.shape[0]} != expected {dim}"
        )

    emb             = normalize(emb)
    embedding_bytes = emb.tobytes()

    idx       = get_mem_index(dim)
    faiss_row = int(idx.ntotal)
    idx.add(emb.reshape(1, -1))

    #FIXED: mark dirty instead of writing the whole index to disk on every insert
    _mem_index_dirty = True

    meta = {
        "conversation_id": req.conversation_id,
        "tags":            req.tags,
        "source":          "client",
    }

    now       = time.time()
    memory_id = str(uuid.uuid4())
    conn      = db()
    conn.execute(
        """INSERT INTO memories
           (memory_id, text, created_at, last_used_at, importance,
            meta_json, faiss_row, embedding)
           VALUES (?,?,?,?,?,?,?,?)""",
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
    """
    Delete a memory by its memory_id from SQLite.
    Note: the FAISS index is not updated live. Call /rebuild_index
    after bulk deletions, or restart the server (index rebuilds on startup
    if embedding BLOBs are present).
    """
    conn = db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        rowcount = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if rowcount == 0:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"ok": True, "deleted": memory_id}

@app.post("/rebuild_index")
def rebuild_index_endpoint():
    """Expose rebuild_faiss_index as an API endpoint."""
    added = rebuild_faiss_index()
    return {"ok": True, "vectors_rebuilt": added}

@app.post("/retrieve_memories")
def retrieve_memories(req: RetrieveReq):
    q = np.array(req.query_embedding, dtype=np.float32)

    dim = get_dim()
    if dim is None:
        return {"memories": []}
    if q.shape[0] != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Query dim {q.shape[0]} != expected {dim}"
        )

    idx = get_mem_index(dim)
    if idx.ntotal == 0:
        return {"memories": []}

    q = normalize(q)
    k = min(req.top_k, int(idx.ntotal))
    if k <= 0:
        return {"memories": []}

    scores, rows = idx.search(q.reshape(1, -1), k)
    scores = scores.flatten().tolist()
    rows   = rows.flatten().tolist()

    valid = [
        (float(score), int(row))
        for score, row in zip(scores, rows)
        if row >= 0 and float(score) >= req.min_score
    ]
    if not valid:
        return {"memories": []}

    #FIXED: single batch query instead of one SELECT per result
    placeholders = ",".join("?" * len(valid))
    row_ids   = [r for _, r in valid]
    score_map = {r: s for s, r in valid}

    conn    = db()
    cur     = conn.cursor()
    db_rows = cur.execute(
        f"SELECT memory_id, text, created_at, last_used_at, importance, "
        f"meta_json, faiss_row FROM memories WHERE faiss_row IN ({placeholders})",
        row_ids
    ).fetchall()

    now = time.time()
    out = []

    for memory_id, text, created_at, last_used_at, importance, meta_json, faiss_row in db_rows:
        meta          = json.loads(meta_json)
        meta["score"] = score_map.get(faiss_row, 0.0)

        if req.conversation_id and meta.get("conversation_id") not in (
            None, req.conversation_id
        ):
            continue

        out.append({
            "memory_id":  memory_id,
            "text":       text,
            "importance": float(importance),
            "meta":       meta,
            "created_at": float(created_at),
        })

        cur.execute(
            "UPDATE memories SET last_used_at=? WHERE memory_id=?",
            (now, memory_id)
        )

    conn.commit()
    conn.close()

    out.sort(key=lambda x: x["meta"]["score"], reverse=True)
    return {"memories": out}

def rebuild_faiss_index() -> int:
    """
    Rebuild the memory FAISS index entirely from the embedding BLOBs
    stored in SQLite. Safe to call after bulk deletes or corruption.
    Returns the number of vectors rebuilt.
    """
    global _mem_index, _mem_index_dirty

    dim = get_dim()
    if dim is None:
        return 0

    idx  = faiss.IndexFlatIP(dim)
    conn = db()
    cur  = conn.cursor()

    #FIXED: SELECT now fetches the embedding BLOB column
    cur.execute(
        "SELECT memory_id, embedding FROM memories ORDER BY faiss_row ASC"
    )
    #FIXED: fetchall() not fetchball()
    rows = cur.fetchall()
    conn.close()

    added = 0
    for memory_id, emb_bytes in rows:
        if emb_bytes is None:
            continue
        #FIXED: emb_bytes is now correctly row[1], not row[0]
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        emb = normalize(emb)
        idx.add(emb.reshape(1, -1))
        added += 1

    _mem_index       = idx
    _mem_index_dirty = True
    flush_mem_index()

    return added

init_db()
