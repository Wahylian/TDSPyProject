"""
Extract network flow features and image data from URLs, yielding records for ML.

This script processes image URLs from a pre-split dataset (split_dataset.csv),
captures associated network flows using NFStream, fetches the image, and yields
a combined record containing the image content, network features, and metadata.
This allows for real-time preprocessing and model training without saving
intermediate files.

Pipeline:
    1. Run create_split.py to generate split_dataset.csv from the raw dataset
    2. Use get_data_stream(split_target='train'|'val'|'test') to process specific splits

Requirements:
    pip install nfstream requests pandas

Windows: must be run as Administrator and requires Npcap (https://npcap.com).
Linux:   must be run as root or with CAP_NET_RAW.

To list available interfaces:
    python -c "import nfstream; print(nfstream.NFStreamer.show_interfaces())"
"""
import os
import socket
import time
import threading
from queue import Empty, Queue
from urllib.parse import urlparse
from typing import Iterator, Dict, Any, Optional, Tuple

import nfstream
import pandas as pd
import requests

# ---- Configuration --------------------------------------------------------
INTERFACE = "Wi-Fi"       # Windows Wi-Fi adapter; change to "Ethernet" or "eth0" as needed
IDLE_TIMEOUT = 10         # Seconds of inactivity before NFStream expires a flow
REQUEST_TIMEOUT = 30      # Per-image HTTP request timeout in seconds
# ---------------------------------------------------------------------------

def _get_dataset_urls(split_target: str = "train") -> Tuple[list, list]:
    """
    Load the pre-split dataset and extract URLs and labels for a specific split.
    
    Args:
        split_target: Which split to load - 'train', 'val', or 'test' (default: 'train')
        
    Returns:
        Tuple of (image_urls, labels) filtered by the specified split
    """
    if split_target not in ["train", "val", "test"]:
        raise ValueError(f"split_target must be 'train', 'val', or 'test', got '{split_target}'")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_csv_path = os.path.join(script_dir, "datasets", "split_dataset.csv")
    
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"split_dataset.csv not found at {split_csv_path}. Run create_split.py first.")
    
    dataset_df = pd.read_csv(split_csv_path)
    print(f"Loaded split dataset from {split_csv_path}")
    print(f"Dataset columns: {list(dataset_df.columns)}")
    
    # Filter by split_target
    split_data = dataset_df[dataset_df["data_split"] == split_target]
    print(f"Filtered {len(split_data)} samples for '{split_target}' split")
    
    # Extract URLs and labels
    valid = split_data[["image_url", "label_numeric"]].dropna()
    return valid["image_url"].tolist(), valid["label_numeric"].tolist()

def _resolve_hostnames(urls: list) -> dict:
    """Pre-resolve hostnames to IP addresses for faster matching."""
    host_to_ip: dict = {}
    for url in urls:
        host = urlparse(url).netloc
        if host and host not in host_to_ip:
            try:
                host_to_ip[host] = socket.gethostbyname(host)
            except socket.gaierror:
                host_to_ip[host] = None
    return host_to_ip

def _start_nfstream_capture(flow_queue: Queue) -> None:
    """Run NFStream in a background thread to capture network flows."""
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

def _drain_flow_queue(flow_queue: Queue) -> list:
    """Wait for flows to expire and collect them from the queue."""
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
    return raw_flows

def get_data_stream(split_target: str = "train") -> Iterator[Dict[str, Any]]:
    """
    Main generator function to process URLs and yield data records from a specific split.

    This function orchestrates the process of:
    1. Loading URLs and labels from the split dataset.
    2. Starting network flow capture.
    3. Fetching each image and its metadata.
    4. Preprocessing the image.
    5. Draining and merging captured network flow data.
    6. Yielding a final, combined record for each URL.
    
    Args:
        split_target: Which split to process - 'train', 'val', or 'test' (default: 'train')
    """
    urls, labels = _get_dataset_urls(split_target=split_target)
    host_to_ip = _resolve_hostnames(urls)
    
    flow_queue: Queue = Queue()
    _start_nfstream_capture(flow_queue)

    # --- Fetch images and collect initial records ---
    download_records = []
    print(f"Fetching {len(urls)} images while capturing flows on '{INTERFACE}'...")
    for i, (url, label) in enumerate(zip(urls, labels), 1):
        host = urlparse(url).netloc
        dst_ip = host_to_ip.get(host)
        record: Dict[str, Any] = {
            "url": url,
            "label": label,
            "dst_ip": dst_ip,
            "http_status": -1,
            "content_length_bytes": 0,
            "response_time_s": -1,
            "image_data": None,
            "flow_features": None,
        }
        try:
            t0 = time.perf_counter()
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            elapsed = round(time.perf_counter() - t0, 3)
            
            record.update({
                "http_status": r.status_code,
                "content_length_bytes": len(r.content),
                "response_time_s": elapsed,
            })

            if r.ok:
                # Store raw image bytes without preprocessing
                record["image_data"] = r.content

        except Exception as e:
            print(f"  Error fetching {url}: {e}")
        
        download_records.append(record)
        if i % 20 == 0:
            print(f"  {i}/{len(urls)}")

    # --- Drain and process network flows ---
    raw_flows = _drain_flow_queue(flow_queue)
    
    # --- Merge flows with download records and yield ---
    download_df = pd.DataFrame(download_records)

    if raw_flows:
        flows_df = pd.DataFrame(raw_flows)
        # Create a map from IP to the first flow matching that IP
        flow_map = {
            flow['dst_ip']: flow for flow in flows_df.to_dict('records')
        }
        
        # Add flow features to the corresponding download record
        for record in download_records:
            if record['dst_ip'] in flow_map:
                record['flow_features'] = flow_map[record['dst_ip']]
    else:
        print("No flows captured — records will not contain flow features.")

    # --- Yield each complete record ---
    print("\n--- Starting Data Stream ---")
    for record in download_records:
        yield record

if __name__ == "__main__":
    # Example of how to consume the data stream with a specific split
    split_choice = "train"  # Change to "val" or "test" as needed
    print(f"Starting feature extraction stream for '{split_choice}' split...")
    
    processed_count = 0
    for data_record in get_data_stream(split_target=split_choice):
        print(f"\n--- Record {processed_count + 1} ---")
        print(f"URL: {data_record['url']}")
        print(f"Label: {data_record['label']}")
        print(f"HTTP Status: {data_record['http_status']}")
        
        # You can now access the preprocessed image data and flow features
        if data_record['image_data'] is not None:
            print(f"Image Data Type: {type(data_record['image_data'])}")
            # print(f"Image Data Shape/Size: {data_record['image_data'].shape}") # If NumPy array
        else:
            print("Image Data: None")
            
        if data_record['flow_features']:
            print("Flow Features: Present")
            # print(data_record['flow_features']) # Uncomment to see details
        else:
            print("Flow Features: None")
            
        processed_count += 1
        
        # In a real scenario, you would feed this data to your model here
        # e.g., train_model(data_record['image_data'], data_record['flow_features'], data_record['label'])

        # Let's just process a few records for this example
        if processed_count >= 5:
            print("\nExample finished after processing 5 records.")
            break
            
    print(f"\nStream finished. Total records processed: {processed_count}")