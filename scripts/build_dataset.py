#!/usr/bin/env python3
import os, json, random, shutil
from datasets import load_dataset
import wave
import numpy as np
import requests
import tarfile


# ── CONFIG ──────────────────────────────────────────────────
OUT_ROOT    = "data/assets"
STREAM_ROOT = "data/stream"
SAMPLES_PER = 2500    # per modality → ~10K total

# helper to copy + record metadata
def write_asset(mod, aid, src, category, tags, title=None):
    """
    Copy `src` into OUT_ROOT/<mod>/<aid><ext> unless it's already there,
    then return the metadata record.
    """
    dst_dir = os.path.join(OUT_ROOT, mod)
    os.makedirs(dst_dir, exist_ok=True)

    ext = os.path.splitext(src)[1]
    dst = os.path.join(dst_dir, f"{aid}{ext}")

    # only copy if src and dst differ
    try:
        src_path = os.path.abspath(src)
        dst_path = os.path.abspath(dst)
        if src_path != dst_path:
            shutil.copyfile(src_path, dst_path)
    except Exception as e:
        # log it and continue; we still want to return the record
        print(f"⚠️  write_asset copy skipped or failed for {src} → {dst}: {e}")

    return {
        "id":       aid,
        "modality": mod,
        "path":     dst,
        "category": category,
        "tags":     tags,
        "title":    title or os.path.basename(src)
    }

# ── 1. TEXT: IMDB reviews ─────────────────────────────────────
def build_text(meta):
    print("▶ Sampling IMDB text…")
    ds = load_dataset("imdb", split="train")  # 25K
    for i, ex in enumerate(ds.shuffle(seed=42).select(range(SAMPLES_PER))):
        path = f"{OUT_ROOT}/text/txt_{i:05d}.txt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(ex["text"])
        meta.append({
            "id":       f"text_{i:05d}",
            "modality": "text",
            "path":     path,
            "category": "imdb_review",
            "tags":     [f"sentiment={ex['label']}"],
            "title":    f"IMDB Review {i}"
        })

# ── 2. IMAGE: Imagenette ─────────────────────────────────────
def build_image(meta):
    print("▶ Downloading & sampling Imagenette…")
    url       = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    archive   = "/tmp/imagenette.tgz"
    extract   = "/tmp/imagenette2-160"
    os.system(f"curl -L {url} -o {archive}")
    os.makedirs(extract, exist_ok=True)
    os.system(f"tar -xzf {archive} -C /tmp")

    train_root = os.path.join(extract, "train")
    all_imgs = []
    for cls in os.listdir(train_root):
        cls_dir = os.path.join(train_root, cls)
        if not os.path.isdir(cls_dir): continue
        for fn in os.listdir(cls_dir):
            all_imgs.append((cls, os.path.join(cls_dir, fn)))

    for i, (cls, path) in enumerate(random.sample(all_imgs, SAMPLES_PER)):
        rec = write_asset("image", f"img_{i:05d}", path, cls, [cls])
        meta.append(rec)

# ── 3. AUDIO: ────────────────────────────────────
def build_audio(meta):
    """
    Generate SAMPLES_PER one-second sine-wave .wav files across several frequencies.
    This is fast, deterministic, and avoids any network stalls.
    """
    print(f"▶ Generating {SAMPLES_PER} synthetic audio clips…")
    out_dir = os.path.join(OUT_ROOT, "audio")
    os.makedirs(out_dir, exist_ok=True)

    # Define 5 distinct frequency classes (in Hz)
    freqs = [220, 440, 880, 1760, 3520]  # A3, A4, A5, A6, A7
    sr    = 16_000    # 16 kHz sampling rate
    dur   = 1.0       # seconds
    t     = np.linspace(0, dur, int(sr * dur), endpoint=False)

    for i in range(SAMPLES_PER):
        f0 = random.choice(freqs)
        samples = (0.5 * np.sin(2 * np.pi * f0 * t) * 32767).astype(np.int16)

        filename = f"aud_{i:05d}.wav"
        path = os.path.join(out_dir, filename)
        # Write WAV
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)            # 16-bit
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

        # Record metadata
        rec = write_asset(
            "audio",
            f"aud_{i:05d}",
            path,
            category=f"sine_{f0}Hz",
            tags=["sine", f"{f0}Hz"]
        )
        meta.append(rec)

    print(f"✔ Built {SAMPLES_PER} synthetic audio assets")


# ── 4. VIDEO: Synthetic color clips ────────────────────────────
def build_video(meta):
    import cv2, numpy as np

    print(f"▶ Synthesizing {SAMPLES_PER} video clips…")
    out_dir = os.path.join(OUT_ROOT, "video")
    os.makedirs(out_dir, exist_ok=True)

    fps      = 5
    dur_sec  = 1
    frames   = fps * dur_sec
    size     = (64, 64)
    codec    = cv2.VideoWriter_fourcc(*"XVID")
    colors   = { "red":(0,0,255), "green":(0,255,0), "blue":(255,0,0),
                 "yellow":(0,255,255), "cyan":(255,255,0) }

    for i in range(SAMPLES_PER):
        cls, bgr = random.choice(list(colors.items()))
        path = os.path.join(out_dir, f"vid_{i:05d}.avi")
        writer = cv2.VideoWriter(path, codec, fps, size)
        frame  = np.full((size[1],size[0],3), bgr, dtype=np.uint8)
        for _ in range(frames):
            writer.write(frame)
        writer.release()

        rec = write_asset("video", f"vid_{i:05d}", path, cls, [cls])
        meta.append(rec)

# ── MAIN: assemble + write metadata + seed stream folder ───────
if __name__=="__main__":
    # prepare dirs
    for m in ("text","image","audio","video"):
        os.makedirs(f"{OUT_ROOT}/{m}", exist_ok=True)
    shutil.rmtree(STREAM_ROOT, ignore_errors=True)
    os.makedirs(STREAM_ROOT, exist_ok=True)

    metadata = []
    build_text(metadata)
    build_image(metadata)
    build_audio(metadata)
    build_video(metadata)

    # write metadata.jsonl
    with open("data/metadata.jsonl","w") as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    # seed 400 random items into stream folder
    for rec in random.sample(metadata, 400):
        shutil.copy(rec["path"], os.path.join(STREAM_ROOT, os.path.basename(rec["path"])))

    print("✅ Built ~10K assets + metadata.jsonl + 400 seeds for streaming.")
