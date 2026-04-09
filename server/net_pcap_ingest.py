import time
import json
import uuid

from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP
from scapy.layers.dns import DNS

from embedder import embed_text
from ml_anomaly import load_model, extract_features, predict
from app import net_db, get_net_index, flush_net_index


def ingest_pcap_file(path: str, capture_id: str) -> int:
    """
    Parse packets from a pcap/pcapng file at `path`
    and store them under `capture_id`.
    """

    packets = rdpcap(path)
    idx = get_net_index(dim=384)

    model = load_model()

    conn = net_db()
    cur = conn.cursor()

    added = 0

    for i, pkt in enumerate(packets):

        meta = {
            "layers": {
                "network": {},
                "transport": {},
                "application": {}
            },
            "packet": {}
        }

        #Packet Layer
        meta["packet"]["timestamp"] = float(pkt.time)
        meta["packet"]["bytes"] = len(pkt)
        meta["packet"]["packet_index"] = i

        #L2
        if ARP in pkt:
            arp = pkt[ARP]
            meta["layers"]["network"] = {
                "type": "ARP",
                "src_ip": arp.psrc,
                "dst_ip": arp.pdst
            }

        #IPv4
        elif IP in pkt:
            ip = pkt[IP]
            meta["layers"]["network"] = {
                "type": "IPv4",
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "protocol_number": ip.proto
            }

        #IPv6
        elif IPv6 in pkt:
            ip = pkt[IPv6]
            meta["layers"]["network"] = {
                "type": "IPv6",
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "protocol_number": ip.nh
            }

        else:
            continue  #skip non network packets

        #Transport Layer
        if TCP in pkt:
            tcp = pkt[TCP]
            meta["layers"]["transport"] = {
                "protocol": "TCP",
                "src_port": tcp.sport,
                "dst_port": tcp.dport
            }

        elif UDP in pkt:
            udp = pkt[UDP]
            meta["layers"]["transport"] = {
                "protocol": "UDP",
                "src_port": udp.sport,
                "dst_port": udp.dport
            }

        elif ICMP in pkt:
            meta["layers"]["transport"] = {
                "protocol": "ICMP"
            }

        #FIXED: DNS block was incorrectly indented inside the ICMP branch,
        if DNS in pkt:
            meta["layers"]["application"]["protocol"] = "DNS"
            dns = pkt[DNS]
            if dns.qd:
                try:
                    meta["layers"]["application"]["query"] = (
                        dns.qd.qname.decode(errors="ignore")
                    )
                except Exception:
                    pass

        #Text for Embedding
        src_ip = meta["layers"]["network"].get("src_ip")
        dst_ip = meta["layers"]["network"].get("dst_ip")
        transport = meta["layers"]["transport"].get("protocol", "N/A")
        src_port = meta["layers"]["transport"].get("src_port")
        dst_port = meta["layers"]["transport"].get("dst_port")

        text = f"{src_ip}:{src_port} → {dst_ip}:{dst_port} [{transport}]"

        #ML anomaly detection
        features = extract_features(meta)
        ml_result = predict(model, features)

        meta["ml"] = ml_result

        #Embedding
        vec = embed_text(text)
        row_id = int(idx.ntotal)
        idx.add(vec.reshape(1, -1))

        #Insert into DB
        cur.execute(
            """
            INSERT INTO net_memories
            (id, capture_id, text, created_at, meta_json, faiss_row)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"{capture_id}:{i}",
                capture_id,
                text,
                meta["packet"]["timestamp"],
                json.dumps(meta),
                row_id,
            ),
        )

        added += 1

    conn.commit()
    conn.close()

    #Mark index dirty and flush to disk
    import app as _app
    _app._net_index_dirty = True
    flush_net_index()

    return added
