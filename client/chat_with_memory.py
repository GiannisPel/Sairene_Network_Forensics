from memory_client import retrieve_memories, add_memory, MEMORY_API, wiki_retrieve, get_server_stats, net_retrieve, net_import_pcap, net_stats
from colorama import Fore, init, Style
from sysinfo import format_neofetch
from animation import animate_once

import json
import requests
import threading
import time
import os
import sys
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import List, Dict, Any


#Config
DISK_PATH = r"S:\\"  #Drive to show in /neofetch
OLLAMA_URL = "http://127.x.x.x:11434/api/chat"
CHAT_MODEL = "qwen2.5:latest"
CONVERSATION_ID = "myproject"
BASE_URL = "http://192.168.x.x:8000"

init(autoreset=True)


#ASCII ART
ASCII_ART = r"""
          =-                                       -=      
         =@@@-                                   -@@@-     
         #@@@@#                                 %@@@@#      
         @@@@@@@:                             -@@@@@@@     
        :@@@@@@@@+                           *@@@@@@@@:    
        -@@@@@@@@@#                         %@@@@@@@@@:     
        :@@@@@@@@@@@                       @@@@@@@@@@@:   
         @@@@@@@@@@+                       +@@@@@@@@@@      
         %@@@@@@@+:                         :*@@@@@@@#      
         +@@@@%-                               =@@@@@=    
         :@@%                                     %@@:       
          +=   :+%%%%+:                 :+%%%%+:   ++     
             =@@@@@@@@@@-             =@@@@@@@@@@-     
            +@@@@@@@@@@@@+           *@@@@@@@@@@@@=      
            @@@@@@:.%@@@@@+         +@@@@@*.=@@@@@@      
            +@@@@@..*@@@@%           %@@@@=..@@@@@+     
              +@@@#=@@@%:    +%@#:    -%@@@=%@@@+    
                               %
"""

ZETA_BANNER = r"""

███████╗███████╗████████╗ █████╗
╚══███╔╝██╔════╝╚══██╔══╝██╔══██╗
  ███╔╝ █████╗     ██║   ███████║
 ███╔╝  ██╔══╝     ██║   ██╔══██║
███████╗███████╗   ██║   ██║  ██║
╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝

""".strip("\n")

NET_BANNER = r"""

       ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
     ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
   ▇▇▇▇                  ▇▇▇▇
 ▇▇▇▇    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇    ▇▇▇▇
▇▇▇    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇    ▇▇
▇    ▇▇▇▇               ▇▇▇    ▇
    ▇▇▇    ▇▇▇▇▇▇▇▇▇▇    ▇▇▇
    ▇    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇    ▇
        ▇▇▇          ▇▇▇
              ▇▇▇▇
              ▇▇▇▇

""".strip("\n")

ERROR_BANNER = r"""

       ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
     ▇▇▇▇▇▇▇▇▇%÷&÷▇▇▇▇▇▇▇▇▇
   ▇▇▇▇       #$±%       ▇▇▇▇
 ▇▇▇▇    ▇▇▇▇▇&÷$#▇▇▇▇▇    ▇▇▇▇
▇▇▇    ▇▇▇▇▇▇▇%±÷&▇▇▇▇▇▇▇    ▇▇▇
▇    ▇▇▇▇     #÷$%     ▇▇▇▇    ▇
    ▇▇▇    ▇▇▇÷&±#▇▇▇    ▇▇▇
    ▇    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇    ▇
        ▇▇▇          ▇▇▇
              /±@%
              %\#:

""".strip("\n")

NET_HELP = """
\n
NETWORK FORENSIC KNOWLEDGE BASE, made by Mpakalogatos.

Net commands:
  /netadd <text>
      Add a manual network note / knowledge chunk.

  /netimp <pcap_or_pcapng_path> [capture_id]
      Import a Wireshark capture into the network RAG store.
      Example: /netimp "C:\Wireshark_captures\This_is_a_test.pcapng"

  /netviz [capture_id] --top-ips/--flow/--anom
      --top-ips: Shows the 10 most communicative IPs with machine
      --flow: Shows the source/port/destination of the top 15 IPs
      --anom: Shows the rate of the anomalies in terminal and in a graph 
      Example: /netviz test.pcapng --top-ips

  /netask [capture_id] | <question>
      Ask questions about imported network data.
      Example: /net ask test1.pcapng | what protocols are present?

  /netstats
      Show quick stats for the current / selected capture.
      
  /nethelp
      Show the Net commands
""".strip("\n")


COLOR_POOL = [
    Fore.CYAN,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.WHITE,
]

PROTO_HINTS = [
    "ARP", "ICMP", "ICMPv6", "IP", "IPv6",
    "TCP", "UDP",
    "DNS", "DHCP", "NTP",
    "HTTP", "HTTPS", "TLS", "QUIC",
    "SSH", "SMTP", "IMAP", "POP",
    "SMB", "NBNS", "LLMNR", "MDNS",
]


def strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s

def color_block(text: str, color=Fore.CYAN) -> str:
    return "\n".join(f"{color}{line}{Style.RESET_ALL}" for line in text.splitlines())

#Helpers
def start_thinking_spinner(label: str = "Zeta: Thinking") -> Any:
    """
    Starts a background spinner that updates the terminal line.
    Returns stop() that stops it and clears the line.
    """
    stop_event = threading.Event()

    def run():
        dots = ["", ".", "..", "..."]
        i = 0
        while not stop_event.is_set():
            sys.stdout.write("\r" + label + dots[i % 4] + "   ")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def stop():
        stop_event.set()
        t.join(timeout=1)
        #Clear the line
        sys.stdout.write("\r" + (" " * 120) + "\r")
        sys.stdout.flush()

    return stop


def ollama_chat_stream(messages: List[Dict[str, str]], model: str, timeout: int = 300) -> str:
    """
    Streams tokens from Ollama and prints them as they arrive.
    Returns the full assistant reply as a string.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    full_text = ""

    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        #Ollama streams NDJSON: one JSON object per line
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            obj = json.loads(line)

            #Typical chunk: {"message":{"role":"assistant","content":"..."}, "done": false}
            msg = obj.get("message")
            if isinstance(msg, dict):
                delta = msg.get("content") or ""
                if delta:
                    print(delta, end="", flush=True)
                    full_text += delta

            if obj.get("done"):
                break

    print()  #Newline after streaming finishes
    return full_text

def extract_protocols_from_text(s: str) -> Counter:
    """
    Extract protocol tokens from Scapy-style summary strings like:
    'Ether / IP / TCP 192.168.x.x:55191 > 192.168.x.x:8006 RA'
    """
    c = Counter()
    if not s:
        return c

    #The cleanest tokens are usually the 'Ether / IP / TCP ...' prefix
    if " / " in s:
        head = s.split("  ", 1)[0]  # up to double-space if present
        parts = [p.strip() for p in head.split("/") if p.strip()]
        for p in parts:
            p2 = p.replace("Ether", "Ethernet").strip()
            # normalize common variants
            if p2.lower() == "ethernet":
                continue
            c[p2] += 1

    #Catch common names anywhere in the string
    upper = s.upper()
    for p in PROTO_HINTS:
        if p in upper:
            c[p] += 1

    return c

def extract_endpoints_from_text(s: str):
    """
    Returns (src_ip, src_port, dst_ip, dst_port) or None.
    """
    if not s:
        return None
    m = IP_PORT_RE.search(s)
    if not m:
        return None
    src = m.group("src")
    dst = m.group("dst")
    srcp = m.group("srcp")
    dstp = m.group("dstp")
    return src, int(srcp) if srcp else None, dst, int(dstp) if dstp else None

IP_PORT_RE = re.compile(
    r"(?P<src>\d{1,3}(?:\.\d{1,3}){3})(?::(?P<srcp>\d{1,5}))?\s*>\s*"
    r"(?P<dst>\d{1,3}(?:\.\d{1,3}){3})(?::(?P<dstp>\d{1,5}))?"
)

def net_viz_top_ips_gui(capture_id: str):
    #Takes the id of the capture, then connects to the subdomain which holds the captures and brings the json data
    r = requests.get(
        f"{BASE_URL}/net/viz/top-ips",
        params={"capture_id": capture_id, "limit": 10},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()

    if not data:
        print("No data returned.")
        return

    ips = [d["ip"] for d in data]
    counts = [d["count"] for d in data]

    fig = px.bar(
        x=ips, y=counts,
        title=f"Top 10 Source IPs - Capture: {capture_id}",
        labels={'x': 'IP Address', 'y': 'Packet Count'},
        template="plotly_dark"
    )
    fig.show() 

def net_viz_flow_gui(capture_id: str):
    r = requests.get(
        f"{BASE_URL}/net/viz/flow",
        params={"capture_id": capture_id},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()

    if not data:
        print("No flow data returned.")
        return

    #Sticking to 20 flows so the graph wont be crowded
    data = data[:20]

    #Getting the data ready for Sankey
    #Finding the individual stations(Source IPs, Ports, Destination IPs) and converting them to string
    all_nodes = list(set(
        [str(d['src']) for d in data] + 
        [f"Port: {d['port']}" for d in data] + 
        [str(d['dst']) for d in data]
    ))
    
    #Creating a mapping (name -> index) cause Plotly needs numbers
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    #Creating the links
    sources = []
    targets = []
    values = []

    for d in data:
        src = str(d['src'])
        port = f"Port: {d['port']}"
        dst = str(d['dst'])
        count = d['count']

        #Connection 1: Source IP -> Port
        sources.append(node_indices[src])
        targets.append(node_indices[port])
        values.append(count)

        #Connection 2: Port -> Destination IP
        sources.append(node_indices[port])
        targets.append(node_indices[dst])
        values.append(count)

    #Generating the graph
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="cyan" #Color for the nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate='Packets: %{value}<extra></extra>'
        )
    )])

    fig.update_layout(
        title_text=f"Network Flow Visualization (Sankey) - {capture_id}",
        font_size=12,
        template="plotly_dark"
    )
    
    fig.show()

def get_anomalies(capture_id):

    r = requests.get(
        f"{BASE_URL}/net/anomalies",
        params={"capture_id": capture_id}
    )

    r.raise_for_status()

    return r.json()

def threat_level(score):

    if score is None:
        return "UNKNOWN"

    if score < -0.20:
        return "CRITICAL"

    if score < -0.12:
        return "HIGH"

    if score < -0.06:
        return "MEDIUM"

    return "LOW"

def print_anomalies(anomalies):

    if not anomalies:
        print("No anomalies found")
        return

    for a in anomalies:

        net = a["layers"]["network"]
        trans = a["layers"]["transport"]
        ml = a.get("ml", {})
        score = ml.get("score")
        level = threat_level(score)

        print("\n⚠ ANOMALY")

        print(
            f'{net.get("src_ip")}:{trans.get("src_port")} → '
            f'{net.get("dst_ip")}:{trans.get("dst_port")}'
        )

        print("score:", ml.get("score"))
        print("threat level:", level)
        print("\nThis tool works as an assistant and may be give you wrong results depening on the dataset its been trained on")

def viz_anomalies_plotly(anomalies, capture_id):
    if not anomalies:
        return

    #Convert to dataframe to be easier
    df_list = []
    for a in anomalies:
        df_list.append({
            "timestamp": a.get("timestamp"), 
            "score": a.get("ml", {}).get("score", 0),
            "src_ip": a["layers"]["network"].get("src_ip"),
            "dst_port": a["layers"]["transport"].get("dst_port"),
            "level": threat_level(a.get("ml", {}).get("score"))
        })
    
    df = pd.DataFrame(df_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    #Initializing colors for every level of threat
    color_map = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "yellow",
        "LOW": "cyan"
    }

    fig = go.Figure()

    #Adding points per threat level
    for level, color in color_map.items():
        sub_df = df[df['level'] == level]
        if not sub_df.empty:
            fig.add_trace(go.Scatter(
                x=sub_df['timestamp'],
                y=sub_df['score'],
                mode='markers',
                name=level,
                marker=dict(color=color, size=10, symbol='diamond'),
                text=[f"IP: {r['src_ip']}<br>Port: {r['dst_port']}" for _, r in sub_df.iterrows()],
                hovertemplate="<b>%{text}</b><br>Score: %{y}<br>Time: %{x}<extra></extra>"
            ))

    #thresholds
    fig.add_hline(y=-0.20, line_dash="dot", line_color="red", annotation_text="Critical Threshold")
    fig.add_hline(y=-0.06, line_dash="dot", line_color="orange")

    fig.update_layout(
        title=f"Anomaly Score Timeline - {capture_id}",
        xaxis_title="Time",
        yaxis_title="Isolation Forest Score (lower is more anomalous)",
        template="plotly_dark",
        hovermode="closest"
    )

    fig.show()

def build_system_messages(memory_block: str, wiki_block: str) -> List[Dict[str, str]]:
    system_text = (
        "You are a local AI assistant running on the user's computer. "
        "The user is speaking directly to you and when the user addresses you, answer accordingly. "
        "Use provided RELEVANT MEMORIES and WIKIPEDIA CONTEXT when they help. "
        "If you are uncertain, say so and ask a brief clarifying question. "
        "Do NOT INVENT facts OR user details OR FAKE memories - DO NOT HALLUCINATE."
        "If network context is incomplete, ask the user for clarification."
        "If a capture shows ambiguous behavior, ask follow-up questions."
    )

    messages = [{"role": "system", "content": system_text}]

    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    if wiki_block:
        messages.append({"role": "system", "content": wiki_block})

    return messages

print(color_block(ZETA_BANNER, Fore.GREEN))
#Chat
def main():
    print(f"Model: {CHAT_MODEL}")
    print(f"Conversation: {CONVERSATION_ID}")
    print("Commands: /remember <text>, /forget <text>, /net, /neofetch, /showanims, /commands | /exit\n")

    history: List[Dict[str, str]] = []

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "exit", "quit"}:
            break
        
        if user_text.lower() in {"/commands"}:
            print("This is the list of the commands.\n")
            print("Commands: /remember <text>, /forget <text>, /net, /neofetch, /showanims | /exit\n")
            continue

        # /neofetch
        if user_text.lower() in {"/neofetch", "neofetch"}:
            animate_once()
            mem_size = None
            wiki_size = None
            try:
                st = get_server_stats()
                mem_size = st.get("memory_store_size_human")
                wiki_size = st.get("wiki_store_size_human")
            except Exception:
                mem_size = None
                wiki_size = None

            print(
                format_neofetch(
                    chat_model=CHAT_MODEL,
                    disk_path=DISK_PATH,
                    memory_db_size=mem_size,
                    wikipedia_size=wiki_size,
                )
            )
            print()
            continue

        # /remember
        if user_text.lower().startswith("/remember "):
            text = user_text[len("/remember "):].strip()
            if text:
                out = add_memory(text, conversation_id=CONVERSATION_ID, importance=0.9, tags=["manual"])
                print(f"Saved memory: {out.get('memory_id')}\n")
            continue

        # /forget (search then delete by id)
        if user_text.lower().startswith("/forget "):
            query = user_text[len("/forget "):].strip()

            r = requests.get(
                f"{MEMORY_API}/search_memories",
                params={"query": query, "conversation_id": CONVERSATION_ID},
                timeout=30,
            )
            r.raise_for_status()
            hits = r.json().get("memories", [])

            if not hits:
                print("No matching memories found.\n")
                continue

            print("\nMatching memories:")
            for i, h in enumerate(hits, start=1):
                print(f"{i}. {h.get('text','')} (ID: {h.get('memory_id','')})")

            choice = input("\nWhich memory number do you want to delete? (or press Enter to cancel): ").strip()
            if not choice:
                print()
                continue
            if not choice.isdigit():
                print("Cancelled.\n")
                continue

            idx = int(choice) - 1
            if not (0 <= idx < len(hits)):
                print("Invalid choice.\n")
                continue

            mem_id = hits[idx].get("memory_id")
            if not mem_id:
                print("Invalid memory id.\n")
                continue

            r = requests.delete(
                f"{MEMORY_API}/delete_memory",
                params={"memory_id": mem_id},
                timeout=30,
            )
            if r.status_code == 200:
                print("Memory deleted!\n")
            else:
                print(f"Failed. HTTP {r.status_code}")
                print(r.text)
                print()
            continue
        
        if user_text.strip().lower() == "/showanims":
            while True:
                print(
                    "\nShow animations:\n"
                    "  1) NET_BANNER\n"
                    "  2) NET_ERROR\n"
                    "  3) cat_logo\n"
                    "  4) ZETA_BANNER\n"
                    "  5) Exit\n"
                )
                choice = input("Choose (1-5): ").strip()
        
                if choice == "1":
                    print(color_block(NET_BANNER, Fore.CYAN))
                elif choice == "2":
                    print(color_block(ERROR_BANNER, Fore.RED))
                elif choice == "3": 
                    print(ASCII_ART)
                elif choice == "4":
                    print(color_block(ZETA_BANNER, Fore.GREEN))
                elif choice == "5":
                    print("Exiting /showanims")
                    break
                else:
                    print("Not a valid choice ! --SELF DESTRUCT--")
                    break
        
            continue
            
        #net analyze
        
        cmd = user_text.strip()
        cmd_l = cmd.lower()
        
        #NET COMMANDS (order matters)
        
        cmd = user_text.strip()
        cmd_l = cmd.lower()
        
        if user_text.lower().startswith("/netask "):
            raw = user_text[len("/netask "):].strip()
        
            capture_id = None
            question = raw
        
            #Allow: "/net ask test1.pcapng | what protocols are present?"
            if "|" in raw:
                left, right = raw.split("|", 1)
                capture_id = left.strip() or None
                question = right.strip()
        
            if not question:
                print("Usage: /netvizask [capture_id |] <question>\n")
                continue
        
            #Retrieve MORE than you show.
            user_top_k = 25
            retrieve_top_k = max(80, user_top_k * 5)
        
            hits = net_retrieve(
                question,
                capture_id=capture_id,
                top_k=retrieve_top_k,
                min_score=0.0
            )
        
            print(f"\nDEBUG net hits: {len(hits)}")
            for h in hits[:5]:
                print(f"- {h['score']:.3f} {h.get('capture_id')} :: {h.get('text','')[:120]}")
            print()
        
            if not hits:
                print("No network knowledge found.\n")
                hits = net_retrieve(question, capture_id=capture_id, top_k=retrieve_top_k, min_score=-1.0)
                continue
        
            #Deterministic extraction from retrieved packet summaries
            proto_counts = Counter()
            ip_counts = Counter()
            flow_counts = Counter()
            port_counts = Counter()
        
            for h in hits:
                t = h.get("text", "")
                proto_counts.update(extract_protocols_from_text(t))
        
                ep = extract_endpoints_from_text(t)
                if ep:
                    src, srcp, dst, dstp = ep
                    ip_counts[src] += 1
                    ip_counts[dst] += 1
                    flow_counts[(src, dst)] += 1
                    if srcp is not None:
                        port_counts[srcp] += 1
                    if dstp is not None:
                        port_counts[dstp] += 1
        
            #Build a compact summary that the LLM can reliably answer from
            top_protocols = proto_counts.most_common(12)
            top_ips = ip_counts.most_common(10)
            top_flows = flow_counts.most_common(8)
            top_ports = port_counts.most_common(10)
        
            summary_lines = []
            if capture_id:
                summary_lines.append(f"Capture filter: {capture_id}")
        
            if top_protocols:
                summary_lines.append(
                    "Protocols (approx counts from retrieved packets): " +
                    ", ".join(f"{p}={n}" for p, n in top_protocols)
                )
            if top_ips:
                summary_lines.append(
                    "Top IPs (approx): " +
                    ", ".join(f"{ip}({n})" for ip, n in top_ips)
                )
            if top_ports:
                summary_lines.append(
                    "Top ports (approx): " +
                    ", ".join(f"{port}({n})" for port, n in top_ports)
                )
            if top_flows:
                summary_lines.append(
                    "Top flows (src -> dst, approx): " +
                    ", ".join(f"{a} -> {b} ({n})" for (a, b), n in top_flows)
                )
        
            net_summary_block = "NET SUMMARY (computed from retrieved packets):\n" + "\n".join(
                f"- {line}" for line in summary_lines
            )
        
            #Keep only the best packet lines to show the model (avoid flooding it)
            #Cap at 30
            top_for_llm = hits[:30]
            net_context_block = "NETWORK CONTEXT (pcap-derived packet summaries):\n" + "\n".join(
                f"- [{h.get('capture_id','')}] {h.get('text','')}" for h in top_for_llm
            )
        
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a local AI assistant.\n"
                        "You will receive NET SUMMARY (computed) and NETWORK CONTEXT (packet summaries).\n"
                        "Use NET SUMMARY as primary truth for 'what protocols/endpoints/ports are present'.\n"
                        "Use packet summaries for supporting details.\n"
                        "If the question is 'what happened', provide a high-level narrative: protocols, endpoints, directionality, and notable events.\n"
                        "Do not invent protocols not present in NET SUMMARY.\n"
                    ),
                },
                {"role": "system", "content": net_summary_block},
                {"role": "system", "content": net_context_block},
            ]
        
            messages.extend(history)
            messages.append({"role": "user", "content": question})
        
            print("\nZeta: ", end="", flush=True)
            reply = ollama_chat_stream(messages, model=CHAT_MODEL)
            print()
        
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": reply})
            continue

        if cmd_l.startswith("/netviz"):
            parts = cmd.split()

            if len(parts) < 3:
                print("Usage: /netviz <capture_id> --top-ips | --flow\n")
                continue

            capture_id = parts[1]
            flag = parts[2]

            try:
                if flag == "--top-ips":
                    net_viz_top_ips_gui(capture_id)

                elif flag == "--flow":
                    net_viz_flow_gui(capture_id)

                elif flag == "--anom":
                    anomalies = get_anomalies(capture_id)

                    #Print on Terminal for fast info
                    print_anomalies(anomalies)

                    #Visualiazation for analysis
                    if anomalies:
                        print(f"\nZeta: Generating anomaly timeline for {capture_id}...")
                        viz_anomalies_plotly(anomalies, capture_id)
                    
                else:
                    print("Unknown flag. Use --top-ips or --flow\n")

            except Exception as e:
                print(f"Visualization failed: {e}\n")

            continue

        # /net import <file path>
        if cmd_l.startswith("/netimp "):
            raw_path = cmd[len("/netimp "):].strip()
            raw_path = strip_quotes(raw_path)
        
            if not raw_path:
                print("Usage: /netimp <path_to_pcap_or_pcapng>\n")
                continue
        
            if not os.path.isfile(raw_path):
                print(f"File not found: {raw_path}\n")
                continue
        
            print("Importing pcap. This can take time (CPU-heavy on the LXC)...")
        
            stop_spinner = start_thinking_spinner("Zeta: Importing")
            try:
                out = net_import_pcap(raw_path, timeout=1800.0)  # 30 minutes
            except requests.exceptions.Timeout:
                stop_spinner()
                print(color_block(ERROR_BANNER, Fore.RED))
                print("Import timed out on the client. The server may still be working.\n")
                continue
            except Exception as e:
                stop_spinner()
                print(color_block(ERROR_BANNER, Fore.RED))
                print(f"Import failed: {e}\n")
                continue
            finally:
                try:
                    stop_spinner()
                except Exception:
                    pass
        
            print(f"Imported: {out}\n")
            continue

        # /netstats
        if cmd_l in {"/netstats"}:
            try:
                out = net_stats()
                print(json.dumps(out, indent=2))
                print()
            except Exception as e:
                print(f"Failed to fetch net stats: {e}\n")
            continue

        # 4) add
        if cmd_l.startswith("/netadd "):
            text = cmd[len("/netadd "):].strip()
            if not text:
                print("Usage: /netadd <text>\n")
                continue
        
            r = requests.post(
                f"{MEMORY_API}/net/add_text",
                json={"text": text, "capture_id": "manual", "tags": ["manual"]},
                timeout=60,
            )
            r.raise_for_status()
            out = r.json()
            print(f"Net memory added. ID: {out.get('memory_id', '(no id returned)')}\n")
            continue
        
        # 5) /net help
        if cmd_l in {"/net", "/net help", "/net/help"}:
            print(color_block(NET_BANNER, Fore.CYAN))
            print(NET_HELP + "\n")
            continue
        
        # 6) net help
        if cmd_l in {"/nethelp"}:
            print(NET_HELP + "\n")
            continue
        
        #Normal chat
        stop_spinner = start_thinking_spinner()

        reply = ""
        try:
            #Retrieval while spinner runs
            wiki_hits = []
            if len(user_text) >= 4:
                wiki_hits = wiki_retrieve(user_text, top_k=6, min_score=0.2)

            wiki_block = ""
            if wiki_hits:
                wiki_block = "WIKIPEDIA CONTEXT:\n" + "\n".join(
                    f"- ({h.get('title','')}) {h.get('text','')}" for h in wiki_hits
                )

            memories = retrieve_memories(
                user_text,
                conversation_id=CONVERSATION_ID,
                top_k=20,
                min_score=0.25,
            )

            memory_block = ""
            if memories:
                memory_block = "RELEVANT MEMORIES:\n" + "\n".join(f"- {m.get('text','')}" for m in memories)

            #Build messages
            messages = build_system_messages(memory_block, wiki_block)
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})

            #Stop spinner before streaming so output is clean
            stop_spinner()

            sys.stdout.write("Zeta: ")
            sys.stdout.flush()

            reply = ollama_chat_stream(messages, model=CHAT_MODEL)

        except KeyboardInterrupt:
            try:
                stop_spinner()
            except Exception:
                pass
            print("\n[Interrupted]\n")
            reply = ""

        finally:
            try:
                stop_spinner()
            except Exception:
                pass

        #Update history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        MAX_TURNS = 12
        if len(history) > MAX_TURNS * 2:
            history = history[-MAX_TURNS * 2:]


if __name__ == "__main__":
    main()
