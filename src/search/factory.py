from pathlib import Path

from .engine import SearchEngine
from src.indexers.bm25_indexer import BM25Indexer
from src.indexers.word2vec_indexer import Word2VecIndexer
from src.indexers.fasttext_indexer import FastTextIndexer


class SearchEngineFactory:
    """Create and cache SearchEngine objects for different indexers"""

    SUPPORTED_INDEXES = ("bm25", "word2vec", "fasttext")

    def __init__(
            self,
            indexes_root: Path,
            preprocessor,
            original_texts: list[str],
            processed_texts: list[str],
            models: dict,
    ):
        """
        :param indexes_root: root directory with index artifacts
        :param preprocessor: text preprocessor to preprocess queries
        :param original_texts: original corpus texts for output
        :param processed_texts: preprocessed corpus texts for searching
        :param models: embedding models if needed, e.g. {"word2vec": w2v_model, "fasttext": ft_model}.
        """
        # check if preprocessed texts and original ones differ
        if len(original_texts) != len(processed_texts):
            raise ValueError("original_texts and processed_texts must have equal length")

        # initialise main vars for indexers
        self.indexes_root = Path(indexes_root)

        # define preprocessor module
        self.preprocessor = preprocessor

        # initialise data
        self.original_texts = original_texts
        self.processed_texts = processed_texts

        # initialise dict with given models
        self.models = models

        # initialise dict for cached indexes
        self._cache: dict[str, SearchEngine] = {}

    def available_indexes(self) -> list[str]:
        """
        :returns: list of supported index names
        """
        return list(self.SUPPORTED_INDEXES)

    def get_engine(self, index_name: str) -> SearchEngine:
        """
        Create SearchEngine instance if cached one doesn't exist
        :param index_name: one of bm25, word2vec, fasttext
        :returns: ready SearchEngine with loaded artifacts
        """
        # check if exists in cache
        key = index_name.lower().strip()
        if key in self._cache:
            return self._cache[key]

        # check if indexer is supported
        if key not in self.SUPPORTED_INDEXES:
            raise ValueError(f'index "{index_name}" is not supported')

        # load indexer
        indexer = self._load_indexer(key)

        # create engine
        engine = SearchEngine(indexer=indexer, preprocessor=self.preprocessor)

        # set texts without rebuilding loaded index artifacts
        engine.original_texts = self.original_texts
        engine.processed_texts = self.processed_texts

        self._cache[key] = engine
        return engine

    def _load_indexer(self, index_name: str):
        """
        Internal function to make indexer ready
        :param index_name: target backend name
        :returns: loaded indexer instance
        """
        if index_name == "bm25":
            return BM25Indexer.load(self.indexes_root / "bm25")

        if index_name == "word2vec":
            model = self.models.get("word2vec")
            if model is None:
                raise ValueError("word2vec model is not provided")
            return Word2VecIndexer.load(self.indexes_root / "word2vec", model=model)

        if index_name == "fasttext":
            model = self.models.get("fasttext")
            if model is None:
                raise ValueError("fasttext model is not provided")
            return FastTextIndexer.load(self.indexes_root / "fasttext", model=model)

        raise ValueError(f'index "{index_name}" is not supported')

    def clear_cache(self) -> None:
        """
        Clear cached engines
        """
        self._cache.clear()
