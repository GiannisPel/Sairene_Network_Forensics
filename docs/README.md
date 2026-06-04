# 🛡️ Project Sairene: AI-Driven Network Forensic Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📒 Overview

**Project Sairene** is a distributed network forensic framework designed to identify stealthy cyberattacks in resource-constrained environments. By combining **Isolation Forest** statistical modeling with a **Heuristic Behavioral Override**, Sairene identifies "Low-and-Slow" exfiltration patterns and stealth scans that typically evade standard detection thresholds.

It combines:

-  **Isolation Forest anomaly detection**
-  **Heuristic Behavioral Overrides**
-  **Retrieval-Augmented Generation (RAG)**
-  **LLM-powered plain-English threat analysis**

Sairene specializes in identifying:

- Low-and-Slow Data Exfiltration  
- Stealth Reconnaissance Scans  
- Beaconing / Callback Malware Traffic  
- Suspicious Flows Hidden Below Threshold Alerts

---

## 🧬 Core Philosophy

- Detect what blends in.  
- Explain what machines ignore.  
- Surface what attackers hide.

---

## 🏗️ Architecture

```mermaid
graph TD

subgraph Client [Analyst Workstation - Windows]
    CLI[chat_with_memory.py]
    MC[memory_client.py]
    LLM[Ollama - Qwen2.5]
    VIZ[Plotly Visualizations]

    CLI --> MC
    CLI --> LLM
    CLI --> VIZ
end

subgraph Server [Memory Engine - Linux / Proxmox]
    API[app.py]
    DB[(SQLite + FAISS)]
    ING[net_pcap_ingest.py]
    FT[flow_tracker.py]
    ML[ml_anomaly.py]
    TRAIN[train_anomaly.py]

    API --> DB
    ING --> FT
    ING --> ML
    ML --> API
    TRAIN --> DB
end

MC --> API
PCAP[Raw PCAP / PCAPNG] --> ING
```
## 📂 Components

### 🧠 Server Side

| File                 | Purpose                                   |
| -------------------- | ----------------------------------------- |
| `app.py`             | FastAPI service, FAISS memory, SQLite API |
| `net_pcap_ingest.py` | Batch packet parser using Scapy           |
| `flow_tracker.py`    | Bidirectional conversation tracker        |
| `l2_tracker.py`      | Heuristic driven and confidence score detection        |
| `ml_anomaly.py`      | Hybrid anomaly detection engine           |
| `train_anomaly.py`   | Offline model retraining                  |

### 👁️ Client Side

| File                  | Purpose                     |
| --------------------- | --------------------------- |
| `chat_with_memory.py` | Main analyst CLI            |
| `memory_client.py`    | HTTP bridge to server       |
| `sysinfo.py`          | Hardware / telemetry module |
| `animation.py`        | Startup UX / persona layer  |

## 🔍 Detection Methodology

**Hybrid Detection Gate**

Sairene uses a three-pass scoring model:

***Pass 1: Statistical Detection***

**Isolation Forest evaluates a 20-feature vector including:**

- Flow duration
- Byte ratios
- Port rarity
- Packet cadence
- Burst patterns

The 1st Pass can detect attacks that have evidence from single packet analysis. For example a XMAS attack and the unusual flag (FIN + PSH + URG combination does not appear in a usual traffic)

***Pass 2: Behavioral Override***

**Rules specifically target:**

*Low-and-Slow Exfiltration:*
- Duration > 30 seconds
- Bitrate < 5000 bps
- Non-standard ports
- Sustained outbound leakage
  
*Stealth Recon:*
- Sparse probing
- Sequential host touches
- Delayed packet cadence
- Low-noise scanning behavior

<p align="center">
  <img src="/docs/screenshots/netask_lownslow_output.png" alt="lowNslow netask example" width="600">
  <br>
  <sup><i>netask Pass 2 example | Low and Slow Exfiltration Attack</i></sup>
</p>

<p align="center">
  <img src="/docs/screenshots/low_n_slow_viz_output.png" alt="STP netviz anom example" width="600">
  <br>
  <sup><i>netviz anom Pass 2 example | Low and Slow Exfiltration Attack</i></sup>
</p>

***Pass 3: Layer 2 Behavioral Summaries***

Pass 3 analyzes traffic that cannot be represented reliably as normal IP flows.  
While Pass 2 groups packets into L3/L4 flows, many infrastructure-level attacks happen at Layer 2 and do not follow TCP/UDP flow semantics.

During this pass, Sairene uses dedicated L2 trackers to aggregate behavior over time and emit `l2_summary` records. These summaries represent incident-level evidence for attacks such as:

- ARP Poisoning
- Gratuitous ARP Storms
- DHCP Starvation
- MAC Flooding
- STP Root Bridge Attacks
- VLAN Hopping / Double-Tagging

<p align="center">
  <img src="/docs/screenshots/STP_netask_output.png" alt="STP netask example" width="600">
  <br>
  <sup><i>netask Passs 3 example | STP Root Bridge Attack</i></sup>
</p>

<p align="center">
  <img src="/docs/screenshots/STP_viz_output.png" alt="STP netviz anom example" width="600">
  <br>
  <sup><i>netviz anom Pass 3 example | STP Root Bridge Attack</i></sup>
</p>

## ⏱️ Bidirectional IAT Tracking

Unlike standard sniffers, Sairene removes ACK-only timing distortion.

This enables accurate detection of:

- Beacon intervals
- Malware sleep-jitter callbacks
- Automated schedulers
- Fake background service traffic

## 💻 Commands

| Command             | Function                |
| ------------------- | ----------------------- |
| `/netimport <file>` | Import PCAP capture     |
| `/netask <query>`   | Query memory with RAG   |
| `/netviz --anom`    | Anomaly timeline        |
| `/netviz --flow`    | Traffic Sankey diagram  |
| `/netviz --top-ips` | Top IP chart            |
| `/netstats`         | Capture summary         |
| `/neofetch`         | Client/server telemetry |

## 🚀 Why Sairene Matters

Traditional IDS systems detect loud attacks.

Modern attackers stay quiet.

Sairene focuses on:

- Subtle behavioral anomalies
- Statistical rarity
- Human-readable explanations
- Lightweight deployment
- Distributed investigation workflows

<div align="center">
🛡️ Sairene

Silent Detection for Quiet Threats.

</div>
