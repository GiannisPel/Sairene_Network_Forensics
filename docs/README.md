# 🛡️ Project Sairene: AI-Driven Network Forensic Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📒 Overview

**Project Sairene** is a distributed network forensic framework designed to identify stealthy cyberattacks in resource-constrained environments. By combining **Isolation Forest** statistical modeling with a **Heuristic Behavioral Override**, Sairene identifies "Low-and-Slow" exfiltration patterns and stealth scans that typically evade standard detection thresholds.

It combines:

- 🌲 **Isolation Forest anomaly detection**
- 🧠 **Heuristic Behavioral Overrides**
- 📚 **Retrieval-Augmented Generation (RAG)**
- 🤖 **LLM-powered plain-English threat analysis**

Sairene specializes in identifying:

- Low-and-Slow Data Exfiltration  
- Stealth Reconnaissance Scans  
- Beaconing / Callback Malware Traffic  
- Suspicious Flows Hidden Below Threshold Alerts

---

## 🧬 Core Philosophy

> Detect what blends in.  
> Explain what machines ignore.  
> Surface what attackers hide.

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

Sairene uses a two-pass scoring model:

***Pass 1: Statistical Detection***

**Isolation Forest evaluates a 20-feature vector including:**

- Flow duration
- Byte ratios
- Port rarity
- Packet cadence
- Burst patterns

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

## 📥 Installation

***Linux / Proxmox Server***

    sudo apt update
    sudo apt install tshark -y
    
    pip install fastapi uvicorn scapy faiss-cpu sentence-transformers joblib scikit-learn
    
    export NET_DATA_DIR="/var/lib/memory_service/net"
    
    uvicorn app:app --host 0.0.0.0 --port 8000

***Windows Analyst Client***

    ollama pull qwen2.5
    
    pip install requests pandas plotly colorama psutil

  Update:
  
    BASE_URL = "http://YOUR_SERVER_IP:8000"

  Run:

    python chat_with_memory.py

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
