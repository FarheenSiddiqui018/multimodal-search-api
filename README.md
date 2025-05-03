# Multimodal Search API

An end-to-end cost-efficient, scalable multi-modal embedding & search system.  

---

## Step 1: Dataset Ingestion

Builds a ~10 K-asset labeled corpus (text, image, audio, video) and loads it into Kafka.

### Prerequisites

- **Python 3.10+**  
- **Homebrew** (for Docker Compose)  
- **Docker Desktop** (or native Kafka via Homebrew)  

### Setup

1. **Clone & enter**  
   ```bash
   git clone https://github.com/FarheenSiddiqui018/multimodal-search-api.git
   cd multimodal-search-api
   ```

2. **Environment variables**  
   Create a `.env` in the project root:
   ```dotenv
   KAFKA_BOOTSTRAP=localhost:9092
   KAFKA_TOPIC=assets
   WATCH_DIR=./data/stream
   ```

3. **Python virtualenv & deps**  
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate

   # Root requirements
   pip install --upgrade pip
   pip install -r requirements.txt

   # Ingestion service requirements
   cd services/ingestion
   pip install -r requirements.txt
   cd ../..
   ```

### Running

1. **Build the dataset**  
   ```bash
   source .venv/bin/activate
   python scripts/build_dataset.py
   ```
   _Generates_:  
   - ~2 500 IMDB text files  
   - ~2 500 Imagenette images  
   - ~2 500 synthetic audio clips  
   - ~2 500 synthetic videos  
   - `metadata.jsonl` & 400 stream seeds in `data/stream/`

2. **Start Kafka & Zookeeper**  
   ```bash
   # Docker Compose (recommended)
   docker compose -f infra/docker-compose.yml up -d

   # or Homebrew
   brew services start zookeeper kafka
   ```
   Ensure topic `assets` exists:
   ```bash
   kafka-topics --create --bootstrap-server localhost:9092 --topic assets
   ```

3. **Batch ingestion**  
   ```bash
   python services/ingestion/batch_ingest.py data/metadata.jsonl
   ```

4. **Streaming ingestion**  
   ```bash
   python services/ingestion/stream_ingest.py
   ```
   Then copy any file into `data/stream/` to “stream” it:
   ```bash
   cp data/assets/text/txt_00000.txt data/stream/
   ```

---

## Step 2: Embedding Generation

Generate multi-modal embeddings in both **batch** and **streaming** modes.  
- **Images, Audio, Video** → ImageBind VISION (audio → mel-spectrogram, video → mid-frame)  
- **Text** → TF-IDF (512-dim vectors)

### Dependencies

```bash
cd services/embedding
pip install --upgrade pip
pip install -r requirements.txt
cd ../..
```

### Batch Embedding

```bash
# quick smoke test (only first 100)
python services/embedding/batch_embed.py --limit 100

# full run
python services/embedding/batch_embed.py
```

You’ll see:

- **VISION** pass (images + audio + video)  
- **TEXT** pass (TF-IDF on text)

Generated `.npy` embeddings land under `data/embeddings/`.

### Streaming Embedding

```bash
# In one shell
source .venv/bin/activate
python services/embedding/stream_embed.py
```
