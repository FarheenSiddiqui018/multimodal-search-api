import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads any FAISS_*_INDEX_PATH / IDMAP_PATH vars

class FaissClient:
    def __init__(self, dim, index_path, idmap_path):
        """
        dim         : embedding dimension
        index_path  : path to .index file
        idmap_path  : path to .json file storing id list
        """
        self.dim         = dim
        self.index_path  = index_path
        self.idmap_path  = idmap_path

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.ids   = json.load(open(self.idmap_path))
            # ensure the loaded index has the expected dimension
            assert self.index.d == self.dim, (
                f"Faiss index dim={self.index.d} != expected {self.dim}"
            )
        else:
            # flat L2 index by default
            self.index = faiss.IndexFlatL2(self.dim)
            self.ids   = []

    def add(self, rec_id: str, vector: np.ndarray):
        """
        rec_id : string ID of the asset
        vector : 1D numpy array of length `dim`
        """
        v = vector.astype("float32").reshape(1, -1)
        assert v.shape[1] == self.dim, (
            f"Vector dim={v.shape[1]} != expected {self.dim}"
        )
        self.index.add(v)
        self.ids.append(rec_id)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        """
        query_vec : 1D numpy array
        returns    : (list_of_ids, list_of_distances)
        """
        q = query_vec.astype("float32").reshape(1, -1)
        D, I = self.index.search(q, top_k)
        ids  = [self.ids[i] for i in I[0]]
        return ids, D[0].tolist()

    def save(self):
        """Persist index and ID map to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.idmap_path, "w") as f:
            json.dump(self.ids, f)
