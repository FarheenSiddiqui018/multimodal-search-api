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

# ── Device & models ────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
vision_model = ImageBindModel().to(device).eval()
if device == "cuda":
    vision_model = vision_model.half()

def prepare_vision_tensor(rec):
    """
    Returns a [1,3,224,224] tensor for image/audio/video via the vision pipeline.
    """
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
    """Batch‐embed all image/audio/video via ImageBind’s VISION."""
    label = "VISION"
    total, t0 = 0, time.time()
    nb = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n→ Embedding {len(records)} vision items in {nb} batches…")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        print(f"  • Batch {i//BATCH_SIZE+1}/{nb} …", end="", flush=True)

        # stack into [B,3,224,224]
        ts = [prepare_vision_tensor(r) for r in batch]
        inp = torch.cat(ts, dim=0)

        t_start = time.time()
        with torch.no_grad():
            embs = vision_model({ModalityType.VISION: inp})[ModalityType.VISION]
        lat = time.time() - t_start

        for idx, rec in enumerate(batch):
            outp = os.path.join(OUT_DIR, f"{rec['id']}.npy")
            np.save(outp, embs[idx].cpu().float().numpy())
            logging.info(f"{label}:{rec['id']} latency={lat:.3f}s")

        total += len(batch)
        print(f" done (lat {lat:.3f}s)")

    avg = (time.time() - t0) / (total or 1)
    print(f"✅ {label} done: {total} items, avg {avg:.3f}s each")

def build_text_vectorizer(records):
    """Train a TF-IDF vectorizer over all text assets."""
    docs = []
    for rec in records:
        raw = open(rec["path"], "rb").read()
        docs.append(raw.decode("utf-8", errors="ignore"))
    vec = TfidfVectorizer(max_features=512)
    vec.fit(docs)
    return vec

def embed_text_group(records, vectorizer):
    """Batch‐embed all text assets via TF-IDF."""
    label = "TEXT"
    total, t0 = 0, time.time()
    nb = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n→ Embedding {len(records)} text items in {nb} batches…")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        print(f"  • Batch {i//BATCH_SIZE+1}/{nb} …", end="", flush=True)

        docs = []
        for rec in batch:
            raw = open(rec["path"], "rb").read()
            docs.append(raw.decode("utf-8", errors="ignore"))

        t_start = time.time()
        embs = vectorizer.transform(docs).toarray()
        lat = time.time() - t_start

        for idx, rec in enumerate(batch):
            outp = os.path.join(OUT_DIR, f"{rec['id']}.npy")
            np.save(outp, embs[idx])
            logging.info(f"{label}:{rec['id']} latency={lat:.3f}s")

        total += len(batch)
        print(f" done (lat {lat:.3f}s)")

    avg = (time.time() - t0) / (total or 1)
    print(f"✅ {label} done: {total} items, avg {avg:.3f}s each")

def batch_embed(limit=None):
    # Load metadata and optionally trim for quick tests
    with open(META_FILE) as f:
        recs = [json.loads(l) for l in f]
    if limit:
        recs = recs[:limit]

    # Split out per‐modality lists
    vision_recs = [r for r in recs if r["modality"] in ("image","audio","video")]
    text_recs   = [r for r in recs if r["modality"] == "text"]

    os.makedirs(OUT_DIR, exist_ok=True)

    if vision_recs:
        embed_vision_group(vision_recs)
    if text_recs:
        tv = build_text_vectorizer(text_recs)
        embed_text_group(text_recs, tv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only embed first N assets (for quick dev)")
    args = p.parse_args()
    batch_embed(limit=args.limit)
