from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from src.indexers.bm25_indexer import BM25Indexer
from src.indexers.fasttext_indexer import FastTextIndexer
from src.indexers.word2vec_indexer import Word2VecIndexer


@dataclass(frozen=True)
class IndexBuildReport:
    """Simple report for index building"""
    index_name: str
    existed: bool
    built: bool
    index_dir: str


class IndexBuildService:
    """Service to check and build indexes and save them"""

    SUPPORTED_INDEXES = ("bm25", "word2vec", "fasttext")

    def __init__(
            self,
            corpus_csv_path: Path,
            indexes_root: Path,
            processed_text_column: str,
            models: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        :param corpus_csv_path: path to preprocessed corpus csv
        :param indexes_root: root dir where indexes will be stored
        :param processed_text_column: column name with preprocessed text (space tokens)
        :param models: embedding models dict, keys: word2vec`` and fasttext
        """
        self.corpus_csv_path = Path(corpus_csv_path)
        self.indexes_root = Path(indexes_root)
        self.processed_text_column = processed_text_column
        self.models = models or {}

    def ensure_indexes(
            self,
            index_names: Optional[list[str]] = None,
            rebuild: bool = False,
    ) -> list[IndexBuildReport]:
        """
        Check indexes and build missing ones
        :param index_names: list of indexes to ensure (default: all supported)
        :param rebuild: if true, rebuild index even if artifacts exist
        :returns: list of reports
        """
        targets = index_names or list(self.SUPPORTED_INDEXES)
        reports: list[IndexBuildReport] = []

        for name in targets:
            key = name.lower().strip()
            if key not in self.SUPPORTED_INDEXES:
                raise ValueError(f'index "{name}" is not supported')

            index_dir = self.indexes_root / key
            existed = self.index_exists(key)
            built = False

            if rebuild or not existed:
                self.build_index(key)
                built = True

            reports.append(
                IndexBuildReport(
                    index_name=key,
                    existed=existed,
                    built=built,
                    index_dir=str(index_dir),
                )
            )

        return reports

    def index_exists(self, index_name: str) -> bool:
        """
        Check if index artifacts exist on disk
        :param index_name: one of bm25``, word2vec``, fasttext
        :returns: true if required artifacts exist
        """
        key = index_name.lower().strip()
        index_dir = self.indexes_root / key

        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            return False

        # check required files for index type
        if key == "bm25":
            return (index_dir / "bm25.pkl").exists()

        if key in ("word2vec", "fasttext"):
            return (index_dir / "doc_vectors.npy").exists() and (index_dir / "tfidf_vectorizer.pkl").exists()

        return False

    def build_index(self, index_name: str) -> None:
        """
        Build a single index and save it to disk
        :param index_name: one of bm25, word2vec, fasttext
        :return: None
        """
        key = index_name.lower().strip()
        if key not in self.SUPPORTED_INDEXES:
            raise ValueError(f'index "{index_name}" is not supported')

        # create root dir if missing
        self.indexes_root.mkdir(parents=True, exist_ok=True)

        # load corpus
        if not self.corpus_csv_path.exists():
            raise FileNotFoundError(f"missing corpus csv: {self.corpus_csv_path}")

        df = pd.read_csv(self.corpus_csv_path)
        if self.processed_text_column not in df.columns:
            raise ValueError(f'missing column "{self.processed_text_column}" in corpus')

        processed_texts = df[self.processed_text_column].fillna("").astype(str).tolist()

        # build and save selected indexer
        index_dir = self.indexes_root / key

        if key == "bm25":
            indexer = BM25Indexer()
            indexer.build_index(processed_texts)
            indexer.save(index_dir)
            return

        if key == "word2vec":
            model = self.models.get("word2vec")
            if model is None:
                raise ValueError("word2vec model is not provided")
            indexer = Word2VecIndexer(model=model)
            indexer.build_index(processed_texts)
            indexer.save(index_dir)
            return

        if key == "fasttext":
            model = self.models.get("fasttext")
            if model is None:
                raise ValueError("fasttext model is not provided")
            indexer = FastTextIndexer(model=model)
            indexer.build_index(processed_texts)
            indexer.save(index_dir)
            return

        raise ValueError(f'index "{index_name}" is not supported')

    def read_index_meta(self, index_name: str) -> dict[str, Any]:
        """
        Read meta.json for an index
        :param index_name: index name
        :returns: parsed meta dict
        """
        key = index_name.lower().strip()
        meta_path = (self.indexes_root / key) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
