# Multimodal Search API

An end-to-end cost-efficient, scalable multi-modal embedding & search system.  

**Step 1: Dataset Ingestion**—building a medium-sized (~10 K assets) labeled corpus and loading it into Kafka via batch and streaming pipelines.
---

## 📦 Prerequisites

- **Python 3.13**  
- **Homebrew** (for Docker Compose)  
- **Docker Desktop** (or native Kafka via Homebrew)  
---

## 🔧 Setup

### 1. Clone & enter

```bash
git clone https://github.com/FarheenSiddiqui018/multimodal-search-api.git
cd multimodal-search-api
````

### 2. Environment variables

Create a file named `.env` in the project root with:

```dotenv
KAFKA_BOOTSTRAP=localhost:9092
KAFKA_TOPIC=assets
WATCH_DIR=./data/stream
```

### 3. Python virtualenv & deps

```bash
# create & activate
python3.13 -m venv .venv
source .venv/bin/activate

# install root dependencies
pip install --upgrade pip
pip install -r requirements.txt

# install ingestion service dependencies
cd services/ingestion
pip install -r requirements.txt
cd ../..
```

---

## 🚀 Dataset Ingestion

### A. Build the dataset

```bash
source .venv/bin/activate
python scripts/build_dataset.py
```

### B. Start Kafka & Zookeeper

#### Option 1: Docker Compose (recommended)

```bash
# from project root
docker compose -f infra/docker-compose.yml up -d
docker compose ps
```

* **Zookeeper** on port 2181
* **Kafka** on port 9092

#### Option 2: Homebrew (native)

```bash
brew install zookeeper kafka
brew services start zookeeper
brew services start kafka
```

---

## 🔄 Ingestion

### 1. Batch ingestion

Publishes **all** assets in `data/metadata.jsonl` to Kafka in one go:

```bash
source .venv/bin/activate
python services/ingestion/batch_ingest.py data/metadata.jsonl
```

### 2. Streaming ingestion

Watches `data/stream/` for new (or existing) files and streams them live:

```bash
source .venv/bin/activate
python services/ingestion/stream_ingest.py
```

