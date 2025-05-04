
# Multimodal Search API

An end-to-end cost-efficient, scalable multi-modal embedding & search system.

**Steps Completed**

---

## 1️⃣ Dataset Ingestion

Build a medium-sized (~10K assets) labeled corpus and load it into Kafka via batch and streaming pipelines.

### 📦 Prerequisites

- **Python 3.10+**  
- **Homebrew** (for Docker Compose & Kafka)  
- **Docker Desktop**

### 🔧 Setup

#### Clone & enter

```bash
git clone https://github.com/FarheenSiddiqui018/multimodal-search-api.git
cd multimodal-search-api
```

#### Environment Variables

Create a `.env` in the project root with:

```dotenv
KAFKA_BOOTSTRAP=localhost:9092
KAFKA_TOPIC=assets
WATCH_DIR=./data/stream
EMB_BATCH_SIZE=32
EMB_TOPIC=embeddings
METRICS_PORT=8001

PG_HOST=localhost
PG_PORT=5432
PG_USER=yourusername
PG_PASSWORD=yourpassword
PG_DATABASE=multimodal
```

#### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🚀 Running Dataset Ingestion

#### Build Dataset

```bash
python scripts/build_dataset.py
```

#### Kafka & Zookeeper

```bash
docker compose -f infra/docker-compose.yml up -d
```

* Zookeeper: `2181`
* Kafka: `9092`
* PostgreSQL: `5432`

---

### 🔄 Ingestion

#### Batch Ingestion

```bash
python services/ingestion/batch_ingest.py data/metadata.jsonl
```

#### Streaming Ingestion

```bash
python services/ingestion/stream_ingest.py
```

---

## 2️⃣ Embedding Generation

Generate embeddings for text, images, audio, and video using ImageBind and TF-IDF (for text).

### Batch Embedding

```bash
python services/embedding/batch_embed.py
```

* Embeddings saved under `data/embeddings/`

### Streaming Embedding

Ensure Kafka is running, then:

```bash
python services/embedding/stream_embed.py
```

---

## 3️⃣ Storage & Indexing

- Embeddings stored using **Faiss** vector database for similarity search.
- Metadata stored in **PostgreSQL**.

### Running PostgreSQL and Faiss Indexing

Ensure PostgreSQL credentials set in `.env`, then:

```bash
docker compose -f infra/docker-compose.yml up -d
```

**Database Schema (Postgres)**

- `id`: UUID (Primary Key)
- `modality`: text
- `asset_path`: text
- `meta`: JSONB

### Indexing embeddings

Embeddings are indexed using Faiss:

- Faiss index stored in `data/`

---

## 📈 Monitoring & Logs

- Logs: `logs/`
- Embedding metrics available via Prometheus at port defined by `METRICS_PORT`.
