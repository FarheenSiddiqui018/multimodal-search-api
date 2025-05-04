# services/query/simple_search.py

import os, sys
import numpy as np

# add project root
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj)

from services.storage.faiss_client import FaissClient
from services.storage.postgres import PostgresClient

# load index
# you need to know dim—just load index and inspect
fa = FaissClient(dim=512)         # or load id map to infer dim
pg = PostgresClient()

# load a query vector (example text query embedding)
q = np.load("data/embeddings/txt_00000.npy")

# search
ids, dists = fa.search(q, top_k=5)
assets = pg.get_assets(ids)

for asset, dist in zip(assets, dists):
    print(f"{asset.id} ({asset.modality}) dist={dist:.4f}")
    print("  metadata:", asset.metadata)
