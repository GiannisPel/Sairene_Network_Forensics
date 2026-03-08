# 🛡️ Project Zeta: AI-Driven Network Forensic Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📒 1. Abstract

Project Zeta is a distributed Client–Server application designed for intelligent network forensic analysis and real-time threat hunting.

The system transforms raw packet captures (PCAP/PCAPNG) into structured, queryable telemetry and augments them with Retrieval-Augmented Generation (RAG) to produce contextualized security insights.

Instead of simply displaying packet data, Zeta introduces a semantic reasoning layer over network traffic, enabling analysts to interpret complex activity patterns with greater clarity and speed.

<p align="center">
  <img src="screenshots/Greetings_Zeta.png" width="800" alt="Zeta says hello">
</p>

## 🏗️ 2. System Architecture

Project Zeta follows a distributed architecture separating ingestion, storage, and reasoning from analyst interaction.

      ┌──────────────┐
      │     PCAP     │
      └──────┬───────┘
             │
             ▼
      ┌──────────────────────────┐
      │ Server (Ingestion & API) │
      └──────┬───────────────────┘
             │
             ▼
      ┌──────────────────────────┐
      │ Structured Network Memory│
      └──────┬───────────────────┘
             │
             ▼
      ┌──────────────────────────┐
      │   LLM Reasoning (RAG)    │
      └──────┬───────────────────┘
             │
             ▼
      ┌──────────────────────────┐
      │ Client & Visualizations  │
      └──────────────────────────┘

Server (Debian / Proxmox):

  - PCAP ingestion
  - Parsing via TShark
  - Structured metadata extraction
  - Storage in SQLite (JSON-enabled)
  - REST API built with FastAPI
  
The server acts as the persistent network memory core.

Client (Windows 11):

  - CLI-based analyst interface
  - Interactive visualizations using Plotly
  - Local LLM orchestration via Ollama

The client is responsible for:

  - 1. Command execution
  - 2. Data visualization
  - 3. Context-aware querying
  - 4. LLM interaction

Database Layer
SQLite with JSON extensions enables:

  - Flexible schema evolution
  - Layer-based packet representation
  - Efficient aggregation queries
  - Structured retrieval for LLM grounding

## 🔑 3. Key Features & Methodology
  ### A. Retrieval-Augmented Generation (RAG)

  Command: /netask <query>
  
  The system does not operate as a generic chatbot.
  
  - Workflow:
  
    - User submits a security-related query.
    - The server retrieves relevant packet records from the database.
    - Structured results are injected into the LLM context.
    - The LLM generates an answer grounded in actual capture data.
    
  This prevents hallucination and ensures that reasoning remains evidence-based.
  
  LLM Model:
  
    Qwen (via Ollama)

  <p align="center">
    <img src="screenshots/netask_example.png" width="900" alt="netask_example">
  </p>

  And with the command: /netviz --top-ips, we can confirm the output of the model:

  <p align="center">
    <img src="screenshots/top_ips_example.png" width="700" alt="netask_example">
  </p>

  ### B. Structured Behavioral Analysis

  Zeta converts packet-level telemetry into structured, queryable representations.
  
  Analytical capabilities include:
  
  - Source-to-destination flow mapping
  - Port usage distribution
  - Traffic concentration analysis
  - High-volume talker identification
  - Protocol distribution tracking
    
  This allows analysts to move from raw packets to interaction-level reasoning.

  ### C. Interactive Visualizations

  Visual abstractions enhance interpretability.
  
  Sankey Diagrams
  
    Source → Port → Destination flow modeling
    Weighted interaction intensity
  
  Dynamic Bar Charts
  
    Top talkers
    Protocol distribution
    Real-time traffic statistics
  
  All visualizations are rendered via Plotly for interactive exploration.

  ### D. Machine Learning Anomaly Detection (Isolation Forest)
  Zeta employs an **Isolation Forest** algorithm to identify outliers in network traffic without requiring pre-labeled attack signatures.
    - **Logic:** The model isolates anomalies based on feature deviations (for example, unusual port-to-IP fan-out, packet frequency).
    - **Threat Scoring:** Each flow is assigned a score. Zeta categorizes these into **CRITICAL, HIGH, MEDIUM,** or **LOW** threat levels based on statistical distance.

## ⌨️ 4. Commands & Usage

  | Command                | Description                                                                 |
  | ---------------------- | --------------------------------------------------------------------------- |
  | `/netimport <file>`    | Parses a PCAP/PCAPNG file and synchronizes it with the server database.     |
  | `/netstats`            | Displays high-level statistics of the current capture.                      |
  | `/netanomalies [capture_id]`| Runs ML detection and generates an interactive anomaly timeline.       |
  | `/netviz --flow`       | Generates an interactive Sankey diagram of network flows.                   |
  | `/netviz --top-ips`    | Generates an interactive Bar graph of the top 10 present IPs.               |
  | `/netask [capture_id]` | Performs a RAG-based forensic analysis using Qwen 2.5.                      |
  | `/neofetch`            | Displays local and server system telemetry.                                 |

  <p align="left">
    <img src="screenshots/net_command.png" width="400" alt="net_command">
  </p>

## 📥 5. Installation & Setup

  - Server Side:

    - 1. Install TShark
         
    - 2. Install dependencies:
         pip install -r server/requirements_server.txt
         
    - 3. Run the FastAPI server:
         uvicorn app:app --host 0.0.0.0 --port 8000

  - Client Side:

    - 1. Install Ollama
         
    - 2. Pull the LLM model:
         ollama pull qwen2.5
         
    - 3. Install client dependencies:
         pip install -r client/requirments_client.txt

# MACHINE LEARNING ALGORITHM FOR NETWORK ANOMALY PER CAPTURE IS IN WORK AND BE PUBLISHED IN THE NEAR FUTURE
