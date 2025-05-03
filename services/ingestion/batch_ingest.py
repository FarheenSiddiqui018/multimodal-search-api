import os, json
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
BOOT = os.getenv("KAFKA_BOOTSTRAP")
TOP  = os.getenv("KAFKA_TOPIC")

producer = KafkaProducer(
    bootstrap_servers=BOOT,
    value_serializer=lambda v: json.dumps(v).encode()
)

def batch_ingest(meta_file):
    with open(meta_file) as f:
        for line in f:
            rec = json.loads(line)
            with open(rec["path"], "rb") as imgf:
                data = imgf.read().hex()
            producer.send(TOP, {"metadata": rec, "data": data})
            print("→ sent", rec["id"])
    producer.flush()

if __name__=="__main__":
    batch_ingest("data/metadata.jsonl")
