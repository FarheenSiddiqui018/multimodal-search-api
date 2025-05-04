#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import argparse

from dotenv import load_dotenv
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# ── ensure project root on PYTHONPATH ─────────────────────
proj_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# ── Core imports ───────────────────────────────────────────
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from services.embedding.transforms import image_transform, audio_image_transform
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Storage clients ─────────────────────────────────────────
from services.storage.postgres import PostgresClient
from services.storage.faiss_client import FaissClient

# ── Env & paths ────────────────────────────────────────────
load_dotenv(os.path.join(proj_root, ".env"))
BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "32"))
META_FILE  = os.path.join(proj_root, "data/metadata.jsonl")
OUT_DIR    = os.path.join(proj_root, "data/embeddings")
LOG_FILE   = os.path.join(proj_root, "logs/batch_embed.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ── Faiss paths per modality ────────────────────────────────
VISION_IDX = os.getenv("FAISS_VISION_INDEX_PATH", "data/faiss_vision.index")
VISION_IDS = os.getenv("FAISS_VISION_IDMAP_PATH",   "data/faiss_vision_ids.json")
TEXT_IDX   = os.getenv("FAISS_TEXT_INDEX_PATH",    "data/faiss_text.index")
TEXT_IDS   = os.getenv("FAISS_TEXT_IDMAP_PATH",    "data/faiss_text_ids.json")

# ── Device & model ─────────────────────────────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
vision_model= ImageBindModel().to(device).eval()
if device == "cuda":
    vision_model = vision_model.half()

# ── Globals for storage clients ────────────────────────────
pg_client      = None
faiss_clients  = {}  # keys: "vision", "text"

def prepare_vision_tensor(rec):
    raw = open(rec["path"], "rb").read()
    if rec["modality"] == "image":
        img = Image.open(BytesIO(raw)).convert("RGB")
        t   = image_transform(img)
    elif rec["modality"] == "audio":
        t = audio_image_transform(raw)
    elif rec["modality"] == "video":
        import cv2
        cap = cv2.VideoCapture(rec["path"])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = length // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Couldn’t read frame from {rec['path']}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t   = image_transform(img)
    else:
        raise ValueError(f"Unknown vision modality {rec['modality']}")
    t = t.to(device)
    if device == "cuda":
        t = t.half()
    return t.unsqueeze(0)

def embed_vision_group(records):
    global pg_client, faiss_clients

    label = "VISION"
    total, start_all = 0, time.time()
    nb = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n→ Embedding {len(records)} vision items in {nb} batches…")

    for batch_idx in range(nb):
        i     = batch_idx * BATCH_SIZE
        batch = records[i : i + BATCH_SIZE]
        print(f"  • Batch {batch_idx+1}/{nb} …", end="", flush=True)

        ts  = [prepare_vision_tensor(r) for r in batch]
        inp = torch.cat(ts, dim=0)

        t0 = time.time()
        with torch.no_grad():
            embs = vision_model({ModalityType.VISION: inp})[ModalityType.VISION]
        lat = time.time() - t0

        # initialize clients on first batch
        if pg_client is None:
            pg_client = PostgresClient()
        if "vision" not in faiss_clients:
            dim = embs.shape[1]
            faiss_clients["vision"] = FaissClient(dim, VISION_IDX, VISION_IDS)

        # save, upsert, index, log
        for idx, rec in enumerate(batch):
            vec  = embs[idx].cpu().float().numpy()
            outp = os.path.join(OUT_DIR, f"{rec['id']}.npy")
            np.save(outp, vec)

            pg_client.upsert_asset(rec)
            faiss_clients["vision"].add(rec["id"], vec)

            logging.info(f"{label}:{rec['id']} latency={lat:.3f}s")

        total += len(batch)
        print(f" done (lat {lat:.3f}s)")

    avg = (time.time() - start_all) / (total or 1)
    print(f"✅ {label} done: {total} items, avg {avg:.3f}s each")

def build_text_vectorizer(records):
    docs = []
    for rec in records:
        raw = open(rec["path"], "rb").read()
        docs.append(raw.decode("utf-8", errors="ignore"))
    vec = TfidfVectorizer(max_features=512)
    vec.fit(docs)
    return vec

def embed_text_group(records, vectorizer):
    global pg_client, faiss_clients

    label = "TEXT"
    total, start_all = 0, time.time()
    nb = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n→ Embedding {len(records)} text items in {nb} batches…")

    # init clients if not already
    if pg_client is None:
        pg_client = PostgresClient()
    if "text" not in faiss_clients:
        dim = vectorizer.max_features
        faiss_clients["text"] = FaissClient(dim, TEXT_IDX, TEXT_IDS)

    for batch_idx in range(nb):
        i     = batch_idx * BATCH_SIZE
        batch = records[i : i + BATCH_SIZE]
        print(f"  • Batch {batch_idx+1}/{nb} …", end="", flush=True)

        docs = [
            open(rec["path"], "rb").read().decode("utf-8", errors="ignore")
            for rec in batch
        ]

        t0   = time.time()
        embs = vectorizer.transform(docs).toarray()
        lat  = time.time() - t0

        for idx, rec in enumerate(batch):
            vec  = embs[idx]
            outp = os.path.join(OUT_DIR, f"{rec['id']}.npy")
            np.save(outp, vec)

            pg_client.upsert_asset(rec)
            faiss_clients["text"].add(rec["id"], vec)

            logging.info(f"{label}:{rec['id']} latency={lat:.3f}s")

        total += len(batch)
        print(f" done (lat {lat:.3f}s)")

    avg = (time.time() - start_all) / (total or 1)
    print(f"✅ {label} done: {total} items, avg {avg:.3f}s each")

def batch_embed(limit=None):
    with open(META_FILE) as f:
        recs = [json.loads(l) for l in f]
    if limit:
        recs = recs[:limit]

    vision_recs = [r for r in recs if r["modality"] in ("image","audio","video")]
    text_recs   = [r for r in recs if r["modality"] == "text"]

    os.makedirs(OUT_DIR, exist_ok=True)

    if vision_recs:
        embed_vision_group(vision_recs)
    if text_recs:
        tv = build_text_vectorizer(text_recs)
        embed_text_group(text_recs, tv)

    # save both indexes
    for client in faiss_clients.values():
        client.save()
    print("✅ Saved all Faiss indexes and id maps")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only embed first N assets (for quick dev)")
    args = p.parse_args()
    batch_embed(limit=args.limit)
