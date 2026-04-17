import time
from collections import defaultdict
import numpy as np

FLOW_TIMEOUT_SECONDS = 120.0
MIN_FLOW_PACKETS     = 5
MAX_IAT_SAMPLES      = 100

TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_ACK = 0x10


class FlowTracker:
    def __init__(self):
        self._flows          = {}
        self._src_dst_ports  = defaultdict(set)
        self._completed      = {}

    def _make_flow_key(self, src_ip, dst_ip, src_port, dst_port, proto):
        """Bidirectional key — A->B and B->A map to the same flow."""
        if (src_ip, src_port) <= (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    def update(self, meta: dict) -> dict:
        network   = meta.get("layers", {}).get("network",   {})
        transport = meta.get("layers", {}).get("transport", {})
        packet    = meta.get("packet", {})

        src_ip    = network.get("src_ip",  "0.0.0.0")
        dst_ip    = network.get("dst_ip",  "0.0.0.0")
        src_port  = int(transport.get("src_port", 0) or 0)
        dst_port  = int(transport.get("dst_port", 0) or 0)
        proto     = int(network.get("protocol_number", 0))
        pkt_size  = int(packet.get("bytes", 0))
        ts        = float(packet.get("timestamp", time.time()))
        tcp_flags = int(transport.get("tcp_flags", 0) or 0)

        if dst_port > 0:
            self._src_dst_ports[src_ip].add(dst_port)

        flow_key = self._make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)

        if flow_key not in self._flows:
            self._flows[flow_key] = {
                "count":        0,
                "bytes":        0,
                "start":        ts,
                "last":         ts,
                "iats":         [],
                "last_fwd_ts":  None,
                "syn_count":    0,
                "ack_count":    0,
                "fin_seen":     False,
                "rst_seen":     False,
                "fwd_count":    0,
                "bwd_count":    0,
                "fwd_bytes":    0,
                "bwd_bytes":    0,
                "is_forward":   None,
                "last_meta":    None,
                #FIXED Track whether a SYN-ACK was ever seen in this flow.
                #A SYN-ACK means the target port responded and the handshake at least started to complete in this case this is NOT a stealth scan.
                #syn_only_flow = not syn_ack_seen is exported in _compute_stats.
                "syn_ack_seen": False,
            }

        f = self._flows[flow_key]

        # Direction detection — first packet sets the forward direction
        if f["is_forward"] is None:
            f["is_forward"] = (src_ip, src_port)

        if (src_ip, src_port) == f["is_forward"]:
            direction = "fwd"
            f["fwd_count"] += 1
            f["fwd_bytes"] += pkt_size
        else:
            direction = "bwd"
            f["bwd_count"] += 1
            f["bwd_bytes"] += pkt_size

        #Forward-only IAT — ACK packets from receiver dont distort the list
        if direction == "fwd":
            if f["last_fwd_ts"] is not None:
                iat = ts - f["last_fwd_ts"]
                if iat >= 0 and len(f["iats"]) < MAX_IAT_SAMPLES:
                    f["iats"].append(iat)
            f["last_fwd_ts"] = ts

        f["count"]     += 1
        f["bytes"]     += pkt_size
        f["last"]       = ts
        f["last_meta"]  = meta

        if tcp_flags & TCP_SYN: f["syn_count"] += 1
        if tcp_flags & TCP_ACK: f["ack_count"] += 1
        if tcp_flags & TCP_FIN: f["fin_seen"]   = True
        if tcp_flags & TCP_RST: f["rst_seen"]   = True

        #FIXED Detect SYN-ACK that marks that the target port responded.
        #Checked regardless of direction: on a local LAN capture both directions are visible, so the server's SYN-ACK arrives as a backward packet and proves the handshake was initiated normally.
        #Once set, this flag is never cleared.
        if (tcp_flags & TCP_SYN) and (tcp_flags & TCP_ACK):
            f["syn_ack_seen"] = True

        if (f["fin_seen"] or f["rst_seen"]) and f["count"] >= MIN_FLOW_PACKETS:
            self._finalize_flow(flow_key)

        return self._compute_stats(flow_key, f, src_ip)

    def _compute_stats(self, flow_key, f, src_ip) -> dict:
        duration = max(f["last"] - f["start"], 1e-6)
        iats     = f["iats"]
        iat_mean = float(np.mean(iats)) if len(iats) >= 2 else 0.0
        iat_std  = float(np.std(iats))  if len(iats) >= 3 else 0.0

        return {
            "count":            f["count"],
            "bytes":            f["bytes"],
            "duration":         duration,
            "pps":              f["count"] / duration,
            "bps":              f["bytes"] / duration,
            "iat_mean":         iat_mean,
            "iat_std":          iat_std,
            "fwd_ratio":        f["fwd_count"] / (f["bwd_count"] + 1),
            "unique_dst_ports": len(self._src_dst_ports.get(src_ip, set())),
            "syn_ack_ratio":    f["syn_count"] / (f["ack_count"] + 1),
            "fwd_packets":      f["fwd_count"],
            "bwd_packets":      f["bwd_count"],
            #FIXED syn_only_flow: True when no SYN-ACK was ever spotted
            #Used by ml_anomaly._explain() to confirm a half open/stealth scan
            #pattern in mature flows, replacing the inaccurate per-packet
            #SYNonly check that fired on every normal connection initiation.
            "syn_only_flow":    not f["syn_ack_seen"],
        }

    def _finalize_flow(self, flow_key) -> None:
        if flow_key not in self._flows:
            return
        f = self._flows[flow_key]
        if f["count"] < MIN_FLOW_PACKETS:
            del self._flows[flow_key]
            return
        src_ip = flow_key[0]
        stats  = self._compute_stats(flow_key, f, src_ip)
        stats["last_meta"] = f["last_meta"]
        stats["flow_key"]  = flow_key
        stats["fin_seen"]  = f["fin_seen"]
        stats["rst_seen"]  = f["rst_seen"]
        self._completed[flow_key] = stats
        del self._flows[flow_key]

    def pop_completed_flows(self) -> list:
        items = list(self._completed.items())
        self._completed.clear()
        return items

    def evict_stale(self, current_time: float | None = None) -> int:
        if current_time is None:
            current_time = time.time()
        evicted = 0
        for k in list(self._flows.keys()):
            f = self._flows[k]
            if (current_time - f["last"]) > FLOW_TIMEOUT_SECONDS:
                if f["count"] >= MIN_FLOW_PACKETS:
                    self._finalize_flow(k)
                else:
                    del self._flows[k]
                evicted += 1
        return evicted


    #Finalize all active flows regardless of timeout
    #Then call at end of PCAP ingest to capture long lived connections that never sent FIN/RST where it is exactly the case for mid capture snapshots of ongoing exfiltration.
    def finalize_all(self) -> list:
        for key in list(self._flows.keys()):
            f = self._flows[key]
            if f["count"] >= MIN_FLOW_PACKETS:
                self._finalize_flow(key)
            else:
                del self._flows[key]
        return self.pop_completed_flows()

    @property
    def active_flow_count(self) -> int:
        return len(self._flows)