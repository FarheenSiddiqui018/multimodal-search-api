#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
from dotenv import load_dotenv
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Histogram, start_http_server
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from transforms import image_transform, audio_image_transform
from sklearn.feature_extraction.text import TfidfVectorizer

# ── ensure project root on PYTHONPATH ───────────────────────
proj_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# ── Env & topics ─────────────────────────────────────────────
load_dotenv(os.path.join(proj_root, ".env"))
BOOTSTRAP    = os.getenv("KAFKA_BOOTSTRAP")
INPUT_TOPIC  = os.getenv("KAFKA_TOPIC")
OUTPUT_TOPIC = os.getenv("EMB_TOPIC", "embeddings")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

# ── Logging & metrics ───────────────────────────────────────
log_path = os.path.join(proj_root, "logs/stream_embed.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

start_http_server(METRICS_PORT)
REQ_COUNT = Counter("emb_requests_total", "Count of embed requests", ["modality"])
LATENCY   = Histogram("emb_latency_seconds", "Embedding latency", ["modality"])

# ── Kafka setup ──────────────────────────────────────────────
consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode())
)
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode()
)

# ── Load ImageBind model ────────────────────────────────────
device       = "cuda" if torch.cuda.is_available() else "cpu"
vision_model = ImageBindModel().to(device).eval()
if device == "cuda":
    vision_model = vision_model.half()

# ── Load full metadata & build TF-IDF for text ──────────────
meta_file = os.path.join(proj_root, "data/metadata.jsonl")
metadata_index = {}
text_docs = []
with open(meta_file) as f:
    for line in f:
        r = json.loads(line)
        metadata_index[r["id"]] = r
        if r["modality"] == "text":
            text_docs.append(open(r["path"], "rb").read().decode("utf-8", errors="ignore"))

print(f"Building TF-IDF over {len(text_docs)} text records…")
text_vectorizer = TfidfVectorizer(max_features=512)
text_vectorizer.fit(text_docs)
print("TF-IDF ready.")

def prepare_vision_tensor_from_path(path, modality):
    """Create [1,3,224,224] tensor from local file at path."""
    if modality == "image":
        img = Image.open(path).convert("RGB")
        t   = image_transform(img)
    elif modality == "audio":
        raw = open(path, "rb").read()
        t   = audio_image_transform(raw)
    elif modality == "video":
        cap = cv2.VideoCapture(path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = length // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame from {path}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t   = image_transform(img)
    else:
        return None
    t = t.to(device)
    if device == "cuda":
        t = t.half()
    return t.unsqueeze(0)

def embed_record(msg):
    meta = msg["metadata"]       # contains id & path
    rec_id  = meta["id"]
    rec_path= meta["path"]
    if rec_id not in metadata_index:
        logging.warning(f"Unknown ID on stream: {rec_id}")
        return

    full_meta = metadata_index[rec_id]
    mod = full_meta["modality"]
    REQ_COUNT.labels(mod).inc()

    # VISION branch covers image, audio, video
    if mod in ("image","audio","video"):
        t = prepare_vision_tensor_from_path(rec_path, mod)
        inp = {ModalityType.VISION: t}
        t0 = time.time()
        with torch.no_grad():
            emb = vision_model(inp)[ModalityType.VISION][0].cpu().float().tolist()
        lat = time.time() - t0
        LATENCY.labels(mod).observe(lat)
        out = {"metadata": full_meta, "embedding": emb}

    # TEXT branch via TF-IDF
    elif mod == "text":
        text = open(rec_path, "rb").read().decode("utf-8", errors="ignore")
        t0 = time.time()
        vec = text_vectorizer.transform([text]).toarray()[0].tolist()
        lat = time.time() - t0
        LATENCY.labels(mod).observe(lat)
        out = {"metadata": full_meta, "embedding": vec}

    else:
        return

    logging.info(f"{mod}:{rec_id} latency={lat:.3f}s")
    producer.send(OUTPUT_TOPIC, out)
    print(f"→ embedded {rec_id} ({mod}) in {lat:.3f}s")

if __name__ == "__main__":
    print(f"👂 Streaming embed listening on `{INPUT_TOPIC}`, metrics @ {METRICS_PORT}")
    for m in consumer:
        embed_record(m.value)
