"""
    [FIX 1] SYN-only heuristic gated on `not flow_mature`:
        Every TCP connection starts with a SYN-only packet. Flagging it
        unconditionally labeled every new connection as a stealth scan.
        For immature flows (count < MIN or duration < MIN), SYN-only still
        fires as before. For mature flows, we now use the more accurate
        `syn_only_flow` stat from FlowTracker instead.

    [FIX 2] `syn_only_flow` from FlowTracker used for mature-flow scan detection:
        FlowTracker now tracks whether a SYN-ACK was ever observed in the flow.
        A mature flow with no SYN-ACK response = confirmed half-open/stealth scan.
        This is factually accurate; the previous per-packet check was a guess.
        predict() now accepts flow_stats and forwards it to _explain().

    [FIX 3] Flow-mature behavioral checks run BEFORE always-valid checks:
        Behavioral findings (low-and-slow exfil, beaconing) are more specific
        and dangerous than generic per-packet flags. When present they should
        dominate the reasons list so the LLM sees the most important finding
        first. The previous ordering let SYN-only crowd them out.

    [FIX 4] `attack_type` field added to predict() output:
        A single best-fit classification string derived by priority from the
        reasons list. Gives the LLM a clean categorical signal instead of
        forcing it to infer attack type from a noisy list of raw reason strings.
        Priority order: LOW_AND_SLOW_EXFIL > BEACONING > SYN_FLOOD > PORT_SCAN
        > STEALTH_SCAN > XMAS_SCAN > DNS_TUNNELING > DNS_RECON > ICMP_TUNNELING
        > FRAGMENTATION_EVASION > LOW_TTL > HIGH_RATE_FLOOD > AUTOMATED_TRAFFIC.

    [CALLERS] net_pcap_ingest.py must pass flow_stats to predict():
        predict(model, features, meta=meta, flow_stats=flow_stats)
        The parameter is optional (defaults to None) so existing call sites
        continue to work, but passing flow_stats enables syn_only_flow detection.
"""

import os
import numpy as np
import joblib

MODEL_PATH = os.environ.get("ANOMALY_MODEL_PATH", "anomaly_model.pkl")

N_FEATURES = 20

MIN_FLOW_PACKETS = 5
MIN_IAT_SAMPLES  = 3
MIN_FLOW_DURATION = 5.0

COMMON_PORTS = {80, 443, 8080, 8443, 22, 53, 8006, 8000, 8008, 8009}

#IPs that legitimately open many ephemeral ports as servers. Without this, Proxmox host (192.168.1.50) flags as port scanner
#because it opens 62+ ephemeral ports during normal operation.
_HIGH_PORT_WHITELIST = {
    "192.168.1.50",   #Proxmox host
    "192.168.1.125",  #Memory service container
}


def load_model():
    return joblib.load(MODEL_PATH)


#Feature extraction

def extract_features(meta: dict, flow_stats: dict | None = None) -> list:
    network   = meta.get("layers", {}).get("network",   {})
    transport = meta.get("layers", {}).get("transport", {})
    app       = meta.get("layers", {}).get("application", {})
    packet    = meta.get("packet", {})

    protocol      = int(network.get("protocol_number", 0))
    src_port      = int(transport.get("src_port", 0) or 0)
    dst_port      = int(transport.get("dst_port", 0) or 0)
    packet_size   = int(packet.get("bytes", 0))
    tcp_flags     = int(transport.get("tcp_flags", 0) or 0)
    ttl           = int(network.get("ttl", 0) or 0)
    is_fragment   = int(bool(network.get("is_fragment", False)))

    header_size   = int(packet.get("header_size", 0))
    payload_bytes = max(0, packet_size - header_size)
    payload_ratio = (payload_bytes / packet_size) if packet_size > 0 else 0.0

    dns_query_length = 0
    dns_answer_count = 0
    dns_is_response  = 0

    if app.get("protocol") == "DNS":
        dns_query_length = int(app.get("query_length", 0) or 0)
        dns_answer_count = int(app.get("answer_count", 0) or 0)
        dns_is_response  = int(bool(app.get("is_response", False)))

    flow_packet_count = 0
    flow_bytes_total  = 0
    flow_duration     = 0.0
    flow_pps          = 0.0
    flow_bps          = 0.0
    iat_mean          = 0.0
    iat_std           = 0.0
    unique_dst_ports  = 0
    syn_ack_ratio     = 0.0

    if flow_stats:
        flow_packet_count = int(flow_stats.get("count",            0))
        flow_bytes_total  = int(flow_stats.get("bytes",            0))
        flow_duration     = float(flow_stats.get("duration",       0.0))
        flow_pps          = float(flow_stats.get("pps",            0.0))
        flow_bps          = float(flow_stats.get("bps",            0.0))
        iat_mean          = float(flow_stats.get("iat_mean",       0.0))
        iat_std           = float(flow_stats.get("iat_std",        0.0))
        unique_dst_ports  = int(flow_stats.get("unique_dst_ports", 0))
        syn_ack_ratio     = float(flow_stats.get("syn_ack_ratio",  0.0))

    return [
        protocol,           #0
        src_port,           #1
        dst_port,           #2
        packet_size,        #3
        tcp_flags,          #4
        ttl,                #5
        is_fragment,        #6
        payload_ratio,      #7
        dns_query_length,   #8
        dns_answer_count,   #9
        dns_is_response,    #10
        flow_packet_count,  #11
        flow_bytes_total,   #12
        flow_duration,      #13
        flow_pps,           #14
        flow_bps,           #15
        iat_mean,           #16
        iat_std,            #17
        unique_dst_ports,   #18
        syn_ack_ratio,      #19
    ]



#Prediction

#For flow summary records, strong heuristic signals can promote an anomaly even when IF score is above threshold but ONLY when _is_strong_signal() confirms the timing data is meaningful
#prevents broken-timestamp captures from generating noise
"""
Returns a dict with keys:
        anomaly     — bool
        score       — float (IsolationForest decision score)
        reasons     — list[str], priority-ordered (most specific first)
        attack_type — str, single best-fit classification  [FIX 4]
"""
def predict(
    model,
    features: list,
    meta: dict | None = None,
    flow_stats: dict | None = None,   #FIXED added now enables syn_only_flow detection
) -> dict:
    score = float(model.decision_function([features])[0])
    pred  = int(model.predict([features])[0])
    is_anomaly = pred == -1

    #FIXED forward flow_stats so _explain() can access syn_only_flow
    reasons = _explain(features, meta, flow_stats)

    if reasons and not is_anomaly:
        #Only for strong behavioral signals
        if _is_strong_signal(features):
            is_anomaly = True

    if any(r.startswith("Low-and-slow exfiltration pattern") for r in reasons):
        is_anomaly = True

    #Suppress IF-only anomalies with no explainable reason
    if is_anomaly and not reasons:
        is_anomaly = False

    #FIXED derive a single best-fit attack type from the reasons list
    attack_type = _classify_attack_type(reasons)

    return {
        "anomaly":     is_anomaly,
        "score":       score,
        "reasons":     reasons,
        "attack_type": attack_type,
    }

#Returns True only when timing data is reliable AND shows clear behavioral anomaly. Prevents broken-timestamp noise promotion.
"""
Requires: 
    flow_duration > 30s  — long enough to be real sustained activity
        count > 10           enough packets for meaningful statistics
        iat_std > 0          broken timestamps produce iat_std=0 exactly, any nonzero value means real timing data exists
        iat_std < 0.1        regular enough to be automated/scripted
"""
def _is_strong_signal(features: list) -> bool:
    flow_duration = features[13]
    iat_std       = features[17]
    count         = features[11]

    return (
        flow_duration > 30.0 and
        count > 10           and
        iat_std > 0          and
        iat_std < 0.1
    )



#Attack type classification

#Priority order reflects specificity and severity: behavioral/multi-packet attacks (which require flow-level evidence) are more trustworthy than per-packet structural flags
#This gives the LLM a clean categorical signal instead of forcing it to infer the attack class from raw reason strings
def _classify_attack_type(reasons: list[str]) -> str:
    """
    Priority (highest → lowest):
        LOW_AND_SLOW_EXFIL  — behavioral, requires duration + timing + bps
        BEACONING           — behavioral, requires regular timing evidence
        SYN_FLOOD           — flow-level, high syn_ack_ratio
        PORT_SCAN           — structural, many unique destination ports
        STEALTH_SCAN        — half-open confirmed OR SYN-only immature flow
        XMAS_SCAN           — structural flag combination
        DNS_TUNNELING       — abnormal DNS query length
        DNS_RECON           — DNS responses with 0 answers
        ICMP_TUNNELING      — oversized ICMP
        FRAGMENTATION_EVASION — IP fragmentation
        LOW_TTL             — suspicious TTL
        HIGH_RATE_FLOOD     — high pps with real timestamps
        AUTOMATED_TRAFFIC   — regular timing (softer, less certain)
        ANOMALOUS_TRAFFIC   — fallback when no pattern matches
    """
    if not reasons:
        return "ANOMALOUS_TRAFFIC"

    combined = " ".join(reasons).lower()

    if "low-and-slow exfiltration" in combined:
        return "LOW_AND_SLOW_EXFIL"
    if "highly regular timing" in combined:
        return "BEACONING"
    if "syn flood" in combined:
        return "SYN_FLOOD"
    if "port scan" in combined:
        return "PORT_SCAN"
    if "stealth scan" in combined or "half-open" in combined:
        return "STEALTH_SCAN"
    if "xmas scan" in combined:
        return "XMAS_SCAN"
    if "dns" in combined and "tunnel" in combined:
        return "DNS_TUNNELING"
    if "dns" in combined and "nxdomain" in combined:
        return "DNS_RECON"
    if "icmp tunneling" in combined:
        return "ICMP_TUNNELING"
    if "fragment" in combined:
        return "FRAGMENTATION_EVASION"
    if "low ttl" in combined:
        return "LOW_TTL"
    if "high packet rate" in combined:
        return "HIGH_RATE_FLOOD"
    if "regular timing pattern" in combined or "automated" in combined:
        return "AUTOMATED_TRAFFIC"

    return "ANOMALOUS_TRAFFIC"


#Explain with human readable reasons
#Structural change, checks now run in this order [FLOW-MATURE BEHAVIORAL CHECKS] and [ALWAYS-VALID STRUCTURAL CHECKS]
#SYN-only heuristic, now split by flow maturity [Immature flow (not flow_mature)] and  [Mature flow (flow_mature)]
#FIXED `syn_only_flow` from flow_stats: FlowTracker.update() now exports syn_only_flow = not syn_ack_seen. A flow where no SYN-ACK was ever observed is a half-open/stealth scan by definition
def _explain(
    features: list,
    meta: dict | None = None,
    flow_stats: dict | None = None,
) -> list[str]:

    reasons = []


    #Whitelist suppresses known normal traffic types

    _WHITELIST_PREFIXES = ("fe80", "ff02", "224.0.0", "255.255")
    if meta is not None:
        src_ip = meta.get("layers", {}).get("network", {}).get("src_ip", "")
        dst_ip = meta.get("layers", {}).get("network", {}).get("dst_ip", "")
        for prefix in _WHITELIST_PREFIXES:
            if src_ip.startswith(prefix) or dst_ip.startswith(prefix):
                return []
    else:
        src_ip = ""


    #Feature unpacking
  
    protocol         = features[0]
    dst_port         = features[2]
    packet_size      = features[3]
    tcp_flags        = features[4]
    ttl              = features[5]
    is_fragment      = features[6]
    dns_query_length = features[8]
    dns_answer_count = features[9]
    flow_count       = features[11]
    flow_duration    = features[13]
    flow_pps         = features[14]
    flow_bps         = features[15]
    iat_mean         = features[16]
    iat_std          = features[17]
    unique_ports     = features[18]
    syn_ack_ratio    = features[19]

    flow_mature = flow_count >= MIN_FLOW_PACKETS and flow_duration >= MIN_FLOW_DURATION

    #FIXED: Extract syn_only_flow from flow_stats.
    #syn_only_flow=True means no SYN-ACK was ever observed in this flow means either the port never responded or the scan was deliberately half-open.
    syn_only_flow = bool(flow_stats.get("syn_only_flow", False)) if flow_stats else False

    #PRIORITY 1 = FLOW-MATURE BEHAVIORAL CHECKS
    #These run first so the most specific findings head the list.
    #The LLM and attack_type classifier both use the first/highest-priority reason, so behavioral evidence must appear before structural noise.

    if flow_mature:

        #TIMING-GATED CHECKS
        #iat_std > 0 is the key guard, broken timestamps produce iat_std=0 exactly. Any nonzero value means real timing data was recorded
        #iat_std < 0.1 catches 3s±0.1s exfil (produces iat_std≈0.057)
        if iat_std > 0:
            if iat_std < 0.1 and flow_count > 10:
                reasons.append(
                    f"Highly regular timing (iat_std={iat_std:.4f}s, "
                    f"mean={iat_mean:.2f}s) — possible beaconing or exfiltration"
                )
            elif iat_std < 0.5 and flow_count > 10:
                reasons.append(
                    f"Regular timing pattern (iat_std={iat_std:.4f}s) — "
                    f"possible automated traffic"
                )

        #Low-and-slow exfiltration
        #Where duration > 30s (not 5s — web requests can last 5s)
        if (
            flow_duration > 30.0
            and iat_std > 0          #real timestamps
            and 0 < flow_bps < 5_000
            and flow_count >= 10
            and dst_port not in COMMON_PORTS
        ):
            reasons.append(
                f"Low-and-slow exfiltration pattern: {flow_bps:.0f} Bps "
                f"to port {dst_port} over {flow_duration:.0f}s"
            )

        # High packet rate (only meaningful with real timestamps)
        if iat_std > 0 and flow_pps > 1_000:
            reasons.append(f"High packet rate ({flow_pps:.0f} pps) — possible flood")

    #PRIORITY 2 = ALWAYS-VALID STRUCTURAL CHECKS
    #These are per-packet signals, timestamp-independent
    #They run after behavioral checks so the reasons list is already populated with the most specific finding when one exists.

    #TCP flag anomalies
    if tcp_flags != 0:

        #XMAS scan, always suspicious regardless of flow state
        if (tcp_flags & 0x29) == 0x29:
            reasons.append("XMAS scan flags (FIN+PSH+URG)")

        #SYN flood — ratio-based, not rate-based
        if syn_ack_ratio > 10 and flow_count > 20:
            reasons.append(f"SYN flood pattern (ratio={syn_ack_ratio:.1f})")

        if (tcp_flags & 0x02) and not (tcp_flags & 0x10):
            if not flow_mature and syn_only_flow:
                #Path A = immature flow AND no SYN-ACK observed yet => suspicious
                #syn_only_flow=False is not checked here because at the very first
                #SYN packet the SYN-ACK hasnt arrived yet (normal TCP timing)
                #If the handshake later completes, syn_only_flow flips to False and subsequent packets will not retrigger this
                reasons.append("SYN-only packet (possible stealth scan)")
            elif flow_mature and syn_only_flow and flow_count >= 3:
                #Path B = confirmed half-open means handshake never completed.
                #This is a mature flow (enough packets + duration) where no SYN-ACK was ever observed. The connection was intentionally
                #left half-open so that means classic stealth/half-open scan pattern.
                reasons.append(
                    f"Half-open flow ({flow_count} SYN pkts, no SYN-ACK observed) "
                    f"— stealth scan confirmed"
                )

    #Port scanning
    #Threshold 100 to avoid flagging normal behaviour.
    #_HIGH_PORT_WHITELIST suppresses false positives for known high port servers.
    if unique_ports > 100 and src_ip not in _HIGH_PORT_WHITELIST:
        reasons.append(f"Port scan behavior ({unique_ports} unique destination ports)")
    elif unique_ports > 200:
        #Extreme scanning, fire even for whitelisted IPs at this volume
        reasons.append(f"Extreme port scan ({unique_ports} unique destination ports)")

    #DNS anomalies
    if dns_query_length > 100:
        reasons.append(
            f"Unusually long DNS query ({dns_query_length} chars) — possible tunneling"
        )
    if dns_answer_count == 0 and features[10] == 1:
        reasons.append("DNS response with 0 answers — possible NXDOMAIN recon")

    #IP fragmentation
    if is_fragment:
        reasons.append("Fragmented IP packet — possible evasion")

    #Suspicious TTL
    if 0 < ttl < 10:
        reasons.append(f"Very low TTL ({ttl}) — possible spoofed packet")

    #ICMP tunneling
    if protocol == 1 and packet_size > 200:
        reasons.append(f"Large ICMP packet ({packet_size}B) — possible ICMP tunneling")

    return reasons
