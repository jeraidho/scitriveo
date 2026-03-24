from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppPaths:
    """
    Unified project paths relative to root directory

    Default paths:
    - source code: root/src
    - indexes: root/indexes
    - models: root/models
    - csv data: root/csv
    """

    root: Path
    src_dir: Path = field(init=False)
    indexes_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    csv_dir: Path = field(init=False)
    collections_dir: Path = field(init=False)

    # default model files
    word2vec_model_path: Path = field(init=False)
    fasttext_model_path: Path = field(init=False)

    # default corpus files
    corpus_preprocessed_csv_path: Path = field(init=False)
    corpus_raw_csv_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.src_dir = self.root / "src"
        self.indexes_dir = self.root / "indexes"
        self.collections_dir = self.root / "collections"
        self.models_dir = self.root / "models"
        self.csv_dir = self.root / "csv"

        self.word2vec_model_path = self.models_dir / "wiki-news-300d-1M.vec"
        self.fasttext_model_path = self.models_dir / "fasttext_model.bin"

        self.corpus_preprocessed_csv_path = self.csv_dir / "abstracts_lemmatized.csv"
        self.corpus_raw_csv_path = self.csv_dir / "abstracts.csv"


@dataclass
class AppConfig:
    """Main runtime config for app container"""

    paths: AppPaths

    # corpus columns
    original_text_column: str = "abstract"
    processed_text_column: str = "index_text_lemmatised"
    seed_text_column: str = "index_text_lemmatised"

    # startup behavior
    ensure_indexes_on_start: bool = True
    rebuild_indexes_on_start: bool = False
    indexes_to_ensure: tuple[str, ...] = ("bm25", "word2vec", "fasttext")

    # recommendation defaults
    recommendation_index_name: str = "fasttext"
    recommendation_rrf_k: int = 60

