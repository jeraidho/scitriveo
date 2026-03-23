from __future__ import annotations
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from .base import BaseIndexer


class BM25Indexer(BaseIndexer):
    """BM25 index by rank_bm25.BM25Okapi"""

    def __init__(self) -> None:
        self.model: BM25Okapi | None = None

    def build_index(self, texts: list[str]) -> None:
        """
        Builds inverted document index by BM-25
        :param texts: documents
        :return: None
        """
        # tokenise for bm250kapi model (docs are already preprocessed earlier)
        tokenized_corpus = [doc.split() for doc in texts]
        self.model = BM25Okapi(tokenized_corpus)

    def get_scores(self, query_tokens: list[str]) -> dict[int, float]:
        """
        Calculate scores by query tokens (BM-25)
        :param query_tokens: list of query tokens
        :return: doc -> score mapping
        """
        if self.model is None:
            return {}

        scores = self.model.get_scores(query_tokens)
        return {i: float(s) for i, s in enumerate(scores) if s > 0}

    def save(self, out_dir: Path) -> None:
        """
        Save bm25 index artifacts
        :param out_dir: target directory to save index artifacts
        """
        # create target directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # check if index is built inside object instance
        if self.model is None:
            raise ValueError("index is not built yet")

        # dump binary pickle with BM250Kapi model
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # add meta information about created index
        meta = {
            "index_type": "bm25",
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: Path, **kwargs) -> "LibraryBM25Indexer":
        """
        Load bm25 index artifacts
        :param in_dir: target directory with index artifacts
        :returns: loaded index object instance
        """
        instance = cls()

        # target path to index dir check if exists
        # load BM250Kapi index object
        bm25_path = in_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"missing file: {bm25_path}")

        with open(bm25_path, "rb") as f:
            instance.model = pickle.load(f)

        return instance
