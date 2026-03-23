from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class BaseIndexer(ABC):
    @abstractmethod
    def build_index(self, texts: list[str]) -> None:
        """
        Build an internal index from preprocessed documents
        :param texts: list of preprocessed documents
        """

    @abstractmethod
    def get_scores(self, query_tokens: list[str]) -> dict[int, float]:
        """
        Score documents with the query
        :param query_tokens: tokenised query
        :returns: doc_id -> score mapping
        """

    @abstractmethod
    def save(self, out_dir: Path) -> None:
        """
        Save index artifacts to given directory
        :param out_dir: target dir where artifacts will be saved
        """

    @classmethod
    @abstractmethod
    def load(cls, in_dir: Path, **kwargs) -> "BaseIndexer":
        """
        Load index artifacts from given directory
        :param in_dir: target dir with index artifacts
        :returns: loaded index object instance
        """
