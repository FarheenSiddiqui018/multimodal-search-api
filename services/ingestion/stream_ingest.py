# services/ingestion/stream_ingest.py

import os
import time
import glob
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()  # loads KAFKA_BOOTSTRAP, KAFKA_TOPIC, WATCH_DIR

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "assets")
WATCH_DIR       = os.getenv("WATCH_DIR", "./data/stream")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # only handle files, not directories
        if event.is_directory:
            return

        path = event.src_path
        aid  = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except Exception as e:
            print(f"  ⚠️  Couldn’t read {path}: {e}")
            return

        payload = {
            "metadata": {"id": aid, "path": path},
            "data": raw.hex()
        }
        producer.send(KAFKA_TOPIC, payload)
        print(f"→ streamed {aid}")

def process_existing_files(handler):
    """
    On startup, process any files already in WATCH_DIR as if they were just created.
    """
    for file_path in glob.glob(os.path.join(WATCH_DIR, "*")):
        if os.path.isfile(file_path):
            class DummyEvent:
                def __init__(self, src_path):
                    self.src_path = src_path
                    self.is_directory = False
            handler.on_created(DummyEvent(file_path))

if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)

    handler = NewFileHandler()

    # 1) Ingest any files already present
    process_existing_files(handler)

    # 2) Start watching for new files
    observer = Observer()
    observer.schedule(handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"👀 Watching directory for new files: {WATCH_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("🛑 Streaming ingestion stopped.")
