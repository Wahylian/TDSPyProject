"""
Extract network flow features from image URLs using NFStream, labeled for ML.

Requirements:
    pip install nfstream requests pandas

Windows: must be run as Administrator and requires Npcap (https://npcap.com).
Linux:   must be run as root or with CAP_NET_RAW.

To list available interfaces:
    python -c "import nfstream; print(nfstream.NFStreamer.show_interfaces())"
"""
import os
import glob
import socket
import time
import threading
from queue import Empty, Queue
from urllib.parse import urlparse

import nfstream
import pandas as pd
import requests

# ---- Configuration --------------------------------------------------------
INTERFACE = "Wi-Fi"       # Windows Wi-Fi adapter; change to "Ethernet" or "eth0" as needed
IDLE_TIMEOUT = 10         # Seconds of inactivity before NFStream expires a flow
REQUEST_TIMEOUT = 30      # Per-image HTTP request timeout in seconds
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- 1. Locate the downloaded dataset CSV ---------------------------------
csv_files = glob.glob(os.path.join(script_dir, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV found — run downloaddataset.py first.")

dataset_df = pd.read_csv(csv_files[0])
print(f"Loaded {len(dataset_df)} rows  |  columns: {list(dataset_df.columns)}")

valid = dataset_df[["image_url", "label"]].dropna()
urls = valid["image_url"].tolist()
labels = valid["label"].tolist()

# ---- 2. Pre-resolve hostnames → IPs (used to match flows to labels) -------
host_to_ip: dict = {}
for url in urls:
    host = urlparse(url).netloc
    if host and host not in host_to_ip:
        try:
            host_to_ip[host] = socket.gethostbyname(host)
        except socket.gaierror:
            host_to_ip[host] = None

# ---- 3. Start NFStream capture in a background daemon thread --------------
flow_queue: Queue = Queue()

def _run_streamer(interface: str, idle_timeout: int) -> None:
    streamer = nfstream.NFStreamer(
        source=interface,
        statistical_analysis=True,
        idle_timeout=idle_timeout,
        active_timeout=120,
    )
    for flow in streamer:
        try:
            flow_queue.put(flow.to_dict())
        except AttributeError:
            flow_queue.put(vars(flow))

threading.Thread(
    target=_run_streamer, args=(INTERFACE, IDLE_TIMEOUT), daemon=True
).start()

# ---- 4. Fetch images and collect HTTP-level features ----------------------
download_records = []
print(f"Fetching {len(urls)} images while capturing flows on '{INTERFACE}'...")

for i, (url, label) in enumerate(zip(urls, labels), 1):
    host = urlparse(url).netloc
    dst_ip = host_to_ip.get(host)
    try:
        t0 = time.perf_counter()
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        elapsed = round(time.perf_counter() - t0, 3)
        download_records.append({
            "url": url,
            "label": label,
            "dst_ip": dst_ip,
            "http_status": r.status_code,
            "content_length_bytes": len(r.content),
            "response_time_s": elapsed,
        })
    except Exception:
        download_records.append({
            "url": url,
            "label": label,
            "dst_ip": dst_ip,
            "http_status": -1,
            "content_length_bytes": 0,
            "response_time_s": -1,
        })
    if i % 50 == 0:
        print(f"  {i}/{len(urls)}")

# ---- 5. Wait for remaining flows to expire, then drain the queue ----------
wait_s = IDLE_TIMEOUT + 2
print(f"Waiting {wait_s}s for remaining flows to expire...")
time.sleep(wait_s)

raw_flows = []
while True:
    try:
        raw_flows.append(flow_queue.get_nowait())
    except Empty:
        break

print(f"Captured {len(raw_flows)} network flows")

# ---- 6. Merge flow features with labels via destination IP ----------------
download_df = pd.DataFrame(download_records)

if raw_flows:
    flows_df = pd.DataFrame(raw_flows)
    ip_label = (
        download_df.dropna(subset=["dst_ip"])
        .groupby("dst_ip")["label"]
        .first()
        .reset_index()
    )
    result_df = flows_df.merge(ip_label, on="dst_ip", how="inner")
    print(f"{len(result_df)} flows matched to a label")
else:
    print("No flows captured — saving HTTP-level features only.")
    result_df = download_df

# ---- 7. Save to CSV -------------------------------------------------------
out = os.path.join(script_dir, "features.csv")
result_df.to_csv(out, index=False)
print(f"\nSaved → {out}")
print(result_df.head())