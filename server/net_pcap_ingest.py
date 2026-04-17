import json

from scapy.all import rdpcap
from scapy.layers.inet  import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2    import ARP
from scapy.layers.dns   import DNS

from embedder     import get_model
from ml_anomaly   import load_model, extract_features, predict
from flow_tracker import FlowTracker
from app          import net_db, get_net_index, flush_net_index
import app as _app
import numpy as np

#Packets embedded and inserted per batch.
#Keeps RAM flat on the 2GB container regardless of capture size.
BATCH_SIZE = 256


def _parse_packet(pkt, index: int) -> dict | None:
    """
    Extract all layers from a Scapy packet into a meta dict.
    Returns None if the packet has no network layer.
    """
    meta = {
        "layers": {
            "network":     {},
            "transport":   {},
            "application": {},
        },
        "packet": {},
    }

    total_size = len(pkt)
    meta["packet"]["timestamp"]    = float(pkt.time)
    meta["packet"]["bytes"]        = total_size
    meta["packet"]["packet_index"] = index
    meta["packet"]["header_size"]  = 0

    #Network layer

    if ARP in pkt:
        arp = pkt[ARP]
        meta["layers"]["network"] = {
            "type":            "ARP",
            "src_ip":          arp.psrc,
            "dst_ip":          arp.pdst,
            "protocol_number": 0,
            "ttl":             0,
            "is_fragment":     False,
        }
        meta["packet"]["header_size"] = 28

    elif IP in pkt:
        ip = pkt[IP]
        meta["layers"]["network"] = {
            "type":            "IPv4",
            "src_ip":          ip.src,
            "dst_ip":          ip.dst,
            "protocol_number": ip.proto,
            "ttl":             ip.ttl,
            "is_fragment":     bool(ip.flags & 0x1),  #MF flag
        }
        meta["packet"]["header_size"] = ip.ihl * 4

    elif IPv6 in pkt:
        ip = pkt[IPv6]
        meta["layers"]["network"] = {
            "type":            "IPv6",
            "src_ip":          ip.src,
            "dst_ip":          ip.dst,
            "protocol_number": ip.nh,
            "ttl":             ip.hlim,
            "is_fragment":     False,
        }
        meta["packet"]["header_size"] = 40

    else:
        return None  #no network layer, skip

    #Transport layer

    if TCP in pkt:
        tcp = pkt[TCP]
        meta["layers"]["transport"] = {
            "protocol":  "TCP",
            "src_port":  tcp.sport,
            "dst_port":  tcp.dport,
            "tcp_flags": int(tcp.flags),
        }
        meta["packet"]["header_size"] += tcp.dataofs * 4

    elif UDP in pkt:
        udp = pkt[UDP]
        meta["layers"]["transport"] = {
            "protocol":  "UDP",
            "src_port":  udp.sport,
            "dst_port":  udp.dport,
            "tcp_flags": 0,
        }
        meta["packet"]["header_size"] += 8

    elif ICMP in pkt:
        icmp = pkt[ICMP]
        meta["layers"]["transport"] = {
            "protocol":  "ICMP",
            "icmp_type": icmp.type,
            "icmp_code": icmp.code,
            "tcp_flags": 0,
        }
        meta["packet"]["header_size"] += 4

    #Application layer => DNS runs over UDP, checked independently of transport layer
    if DNS in pkt:
        dns = pkt[DNS]
        app_info = {
            "protocol":     "DNS",
            "is_response":  bool(dns.qr),
            "answer_count": int(dns.ancount) if hasattr(dns, "ancount") else 0,
            "query_length": 0,
        }
        if dns.qd:
            try:
                qname = dns.qd.qname.decode(errors="ignore").rstrip(".")
                app_info["query"]        = qname
                app_info["query_length"] = len(qname)
            except Exception:
                pass
        meta["layers"]["application"] = app_info

    return meta


def _build_embed_text(meta: dict) -> str:

    net       = meta["layers"]["network"]
    transport = meta["layers"]["transport"]
    app       = meta["layers"]["application"]

    src_ip    = net.get("src_ip")
    dst_ip    = net.get("dst_ip")
    proto     = transport.get("protocol", "N/A")
    src_port  = transport.get("src_port")
    dst_port  = transport.get("dst_port")
    flags     = transport.get("tcp_flags", 0)
    app_proto = app.get("protocol", "")
    dns_query = app.get("query", "")

    parts = [f"{src_ip}:{src_port} → {dst_ip}:{dst_port} [{proto}]"]

    if flags:
        flag_names = []
        if flags & 0x02: flag_names.append("SYN")
        if flags & 0x10: flag_names.append("ACK")
        if flags & 0x04: flag_names.append("RST")
        if flags & 0x01: flag_names.append("FIN")
        if flags & 0x08: flag_names.append("PSH")
        if flag_names:
            parts.append(f"flags={'+'.join(flag_names)}")

    if app_proto:
        parts.append(app_proto)
    if dns_query:
        parts.append(f"query={dns_query}")

    return " ".join(parts)

#Parse all packets from a .pcap file, then compute enriched metadata + flow stats + ML anomaly scores, embed in batches then store everything under `capture_id`
"""
    Two scoring passes:
    1. Per-packet pass — scores every packet immediately (fast attacks)
    2. Flow-level pass — scores each completed flow as a unit (slow attacks)

    Returns the number of records successfully stored.
"""
def ingest_pcap_file(path: str, capture_id: str) -> int:
    packets      = rdpcap(path)
    model        = load_model()
    tracker      = FlowTracker()
    embed_model  = get_model()

    idx  = get_net_index(dim=384)
    conn = net_db()
    cur  = conn.cursor()

    batch_texts = []
    batch_metas = []
    batch_ids   = []
    batch_ts    = []

    added = 0

    def flush_batch():
        nonlocal added
        if not batch_texts:
            return

        vecs = embed_model.encode(
            batch_texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        start_row = int(idx.ntotal)
        idx.add(vecs)

        rows = [
            (
                batch_ids[j],
                capture_id,
                batch_texts[j],
                batch_ts[j],
                json.dumps(batch_metas[j], ensure_ascii=False),
                start_row + j,
            )
            for j in range(len(batch_texts))
        ]

        cur.executemany(
            """
            INSERT OR IGNORE INTO net_memories
            (id, capture_id, text, created_at, meta_json, faiss_row)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        added += len(rows)

        batch_texts.clear()
        batch_metas.clear()
        batch_ids.clear()
        batch_ts.clear()

    #Pass 1 = Per-packet scoring
    #Catches fast attacks: port scans, SYN floods, DNS tunneling and stores one record per packet with partial flow stats at that moment.
    for i, pkt in enumerate(packets):
        meta = _parse_packet(pkt, i)
        if meta is None:
            continue

        flow_stats    = tracker.update(meta)
        meta["flow"]  = flow_stats
        meta["flow_record_type"] = "packet"

        try:
            features  = extract_features(meta, flow_stats)
            ml_result = predict(model, features, meta=meta, flow_stats=flow_stats)
            meta["ml"] = ml_result
        except Exception:
            meta["ml"] = {"anomaly": False, "score": 0.0, "reasons": [], "trained": False}

        batch_texts.append(_build_embed_text(meta))
        batch_metas.append(meta)
        batch_ids.append(f"{capture_id}:{i}")
        batch_ts.append(meta["packet"]["timestamp"])

        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    #Pass 2 = Flow-level scoring
    #Catches slow attacks: low-and-slow exfiltration, beaconing
    #finalize_all() forces completion of every flow still open at end of capture which is critical for connections that never sent FIN/RST, which is
    #exactly what a mid-capture snapshot of an ongoing exfil looks like. Stores one summary record per completed flow.
    completed_flows = tracker.finalize_all()

    for flow_idx, (flow_key, final_stats) in enumerate(completed_flows):
        src_ip, dst_ip, src_port, dst_port, proto = flow_key

        #Use the last packet's meta as the representative record but replace its flow stats with the complete final stats
        rep_meta = final_stats.pop("last_meta", None)
        if rep_meta is None:
            continue

        rep_meta = dict(rep_meta)  #shallow copy. Dont mutate original
        rep_meta["flow"] = final_stats
        rep_meta["flow_record_type"] = "flow_summary"

        try:
            features  = extract_features(rep_meta, final_stats)
            ml_result = predict(model, features, meta=meta, flow_stats=flow_stats)
            rep_meta["ml"] = ml_result
        except Exception:
            rep_meta["ml"] = {"anomaly": False, "score": 0.0, "reasons": [], "trained": False}

        #Build a descriptive text for the flow summary
        pn = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto, str(proto))
        dur  = final_stats.get("duration", 0)
        pkts = final_stats.get("count", 0)
        bps  = final_stats.get("bps", 0)
        text = (
            f"FLOW {src_ip}:{src_port} → {dst_ip}:{dst_port} [{pn}] "
            f"packets={pkts} duration={dur:.1f}s bps={bps:.0f}"
        )

        ts = rep_meta["packet"]["timestamp"]
        record_id = f"{capture_id}:flow:{flow_idx}"

        batch_texts.append(text)
        batch_metas.append(rep_meta)
        batch_ids.append(record_id)
        batch_ts.append(ts)

        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    conn.close()

    _app._net_index_dirty = True
    flush_net_index()

    return added


def ingest_pcap_file_stream(path: str, capture_id: str):
    packets     = rdpcap(path)
    total       = len(packets)
    model       = load_model()
    tracker     = FlowTracker()
    embed_model = get_model()

    idx  = get_net_index(dim=384)
    conn = net_db()
    cur  = conn.cursor()

    batch_texts = []
    batch_metas = []
    batch_ids   = []
    batch_ts    = []

    added = 0

    #Announce packet count so the client initialises bar with this
    yield (0, total)

    def flush_batch():
        nonlocal added
        if not batch_texts:
            return

        vecs = embed_model.encode(
            batch_texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        start_row = int(idx.ntotal)
        idx.add(vecs)

        rows = [
            (
                batch_ids[j],
                capture_id,
                batch_texts[j],
                batch_ts[j],
                json.dumps(batch_metas[j], ensure_ascii=False),
                start_row + j,
            )
            for j in range(len(batch_texts))
        ]

        cur.executemany(
            """
            INSERT OR IGNORE INTO net_memories
            (id, capture_id, text, created_at, meta_json, faiss_row)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        added += len(rows)

        batch_texts.clear()
        batch_metas.clear()
        batch_ids.clear()
        batch_ts.clear()

    for i, pkt in enumerate(packets):
        meta = _parse_packet(pkt, i)
        if meta is None:
            continue

        flow_stats    = tracker.update(meta)
        meta["flow"]  = flow_stats
        meta["flow_record_type"] = "packet"

        try:
            features  = extract_features(meta, flow_stats)
            ml_result = predict(model, features, meta=meta, flow_stats=flow_stats)
            meta["ml"] = ml_result
        except Exception:
            meta["ml"] = {"anomaly": False, "score": 0.0, "reasons": [], "trained": False}

        batch_texts.append(_build_embed_text(meta))
        batch_metas.append(meta)
        batch_ids.append(f"{capture_id}:{i}")
        batch_ts.append(meta["packet"]["timestamp"])

        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()
            yield (added, total)

    flush_batch()

    #Flow-level pass has same logic as ingest_pcap_file
    completed_flows = tracker.finalize_all()
    for flow_idx, (flow_key, final_stats) in enumerate(completed_flows):
        src_ip, dst_ip, src_port, dst_port, proto = flow_key
        rep_meta = final_stats.pop("last_meta", None)
        if rep_meta is None:
            continue
        rep_meta = dict(rep_meta)
        rep_meta["flow"] = final_stats
        rep_meta["flow_record_type"] = "flow_summary"
        try:
            features  = extract_features(rep_meta, final_stats)
            ml_result = predict(model, features, meta=meta, flow_stats=flow_stats)
            rep_meta["ml"] = ml_result
        except Exception:
            rep_meta["ml"] = {"anomaly": False, "score": 0.0, "reasons": [], "trained": False}
        pn = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto, str(proto))
        dur  = final_stats.get("duration", 0)
        pkts = final_stats.get("count", 0)
        bps  = final_stats.get("bps", 0)
        text = (
            f"FLOW {src_ip}:{src_port} → {dst_ip}:{dst_port} [{pn}] "
            f"packets={pkts} duration={dur:.1f}s bps={bps:.0f}"
        )
        batch_texts.append(text)
        batch_metas.append(rep_meta)
        batch_ids.append(f"{capture_id}:flow:{flow_idx}")
        batch_ts.append(rep_meta["packet"]["timestamp"])
        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()
            yield (added, total)

    flush_batch()

    conn.close()

    _app._net_index_dirty = True
    flush_net_index()
    yield (added, total, capture_id)
