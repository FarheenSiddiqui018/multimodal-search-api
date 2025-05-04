# services/storage/postgres.py

import os
from dotenv import load_dotenv
from sqlalchemy import Column, String, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ─── Load env & build DB URL ───────────────────────────────
load_dotenv()  # reads PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB

DATABASE_URL = (
    f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASS')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}"
    f"/{os.getenv('PG_DB')}"
)

# ─── SQLAlchemy setup ───────────────────────────────────────
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

class Asset(Base):
    __tablename__ = "assets"

    # primary key = embedding ID
    id            = Column(String, primary_key=True, index=True)
    modality      = Column(String, index=True)          # text|image|audio|video
    path          = Column(String, nullable=False)      # local file path
    metadata_json = Column("metadata", JSON, nullable=True)
    # ^ stores all other JSON fields under a column named 'metadata'

# create the table if it doesn't already exist
Base.metadata.create_all(bind=engine)

class PostgresClient:
    def __init__(self):
        self.db = SessionLocal()

    def upsert_asset(self, rec: dict):
        """
        rec: the record dictionary coming from metadata.jsonl or the stream,
             must contain 'id', 'modality', 'path', plus any extra fields.
        """
        # strip out the core columns, everything else goes into metadata_json
        meta = {k: v for k, v in rec.items() if k not in ("id", "modality", "path")}
        asset = Asset(
            id=rec["id"],
            modality=rec["modality"],
            path=rec["path"],
            metadata_json=meta
        )
        self.db.merge(asset)
        self.db.commit()

    def get_assets(self, ids):
        """
        Given a list of IDs, return the corresponding Asset objects in the same order.
        """
        rows = (
            self.db
            .query(Asset)
            .filter(Asset.id.in_(ids))
            .all()
        )
        lookup = {r.id: r for r in rows}
        return [lookup[i] for i in ids if i in lookup]
