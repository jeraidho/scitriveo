from __future__ import annotations
from typing import Any, Optional
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseIndexer


class Word2VecIndexer(BaseIndexer):
    """
    Indexer for word2vec model
    Document vector is a TF-IDF weighted mean of token vectors
    Query score is calculated by cosine similarity between normalised vectors
    """

    def __init__(
        self,
        model: Any,
        tfidf_max_features: Optional[int] = 50_000,
        min_score: float = 0.0,
    ) -> None:
        """
        :param model: a word2vec-like model
        :param tfidf_max_features: maximum for TF-IDF vocabulary size
        :param min_score: minimum score to return
        """
        # initialise model and input
        self.model = model
        self.tfidf_max_features = tfidf_max_features
        self.min_score = min_score

        # create future classes for TF-IDF vectors
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.feature_names: Optional[np.ndarray] = None

        # initialise doc vectors of shape: (n_docs, dim)
        self.doc_vectors: Optional[np.ndarray] = None

    def _get_token_vector(self, token: str) -> Optional[np.ndarray]:
        """
        Internal function to extract vector from model
        :param token: target token
        :return: numpy array with token embedding
        """
        # check if model is not defined
        if not hasattr(self.model, "wv"):
            return None

        # otherwise initialise wv
        wv = self.model.wv

        # check if key -> index mapping exists in word2vec model
        if hasattr(wv, "key_to_index") and token not in wv.key_to_index:
            return None
        try:
            return np.asarray(wv[token], dtype=np.float32)
        except KeyError:
            return None

    def build_index(self, texts: list[str]) -> None:
        """
        Function to create doc index, redefine self.doc_vectors
        :param texts: list of texts to preprocess
        :return: None
        """
        # create empty index if empty text list is given
        if not texts:
            # create doc zeros array with shape of word2vec model vector size
            self.doc_vectors = np.zeros((0, int(getattr(self.model, "vector_size", 0))), dtype=np.float32)

            # create tokenizer
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=str.split, preprocessor=None, lowercase=False, max_features=self.tfidf_max_features
            )
            self.feature_names = np.array([], dtype=str)
            return

        # create tf_idf matrix for given preprocessed texts
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            max_features=self.tfidf_max_features,
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = np.asarray(self.tfidf_vectorizer.get_feature_names_out())

        # create doc vector matrix
        dim = int(getattr(self.model, "vector_size"))
        doc_vectors = np.zeros((len(texts), dim), dtype=np.float32)

        # for each doc add vector of each token
        for doc_id in range(len(texts)):
            row = tfidf_matrix.getrow(doc_id)
            vec = np.zeros(dim, dtype=np.float32)
            weight_sum = 0.0

            # use only tokens with non-zero tf-idf weight
            for feature_idx, weight in zip(row.indices, row.data):
                token = str(self.feature_names[feature_idx])
                token_vec = self._get_token_vector(token)
                if token_vec is None:
                    continue
                vec += float(weight) * token_vec
                weight_sum += float(weight)

            if weight_sum > 0:
                vec /= weight_sum

            # normalise, cosine similarity is dot product
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm

            doc_vectors[doc_id] = vec

        self.doc_vectors = doc_vectors

    def _build_query_vector(self, query_tokens: list[str]) -> Optional[np.ndarray]:
        """
        Internal function to create vector for query tokens
        :param query_tokens: list of target words from query
        :return: None (if query vector is empty) or target query vector (numpy-array)
        """
        # if something is not defined
        if self.tfidf_vectorizer is None or self.feature_names is None:
            return None
        if not query_tokens:
            return None

        # join list of tokens
        q_text = " ".join(query_tokens)

        # get tfidf vector for query
        q_tfidf = self.tfidf_vectorizer.transform([q_text])
        row = q_tfidf.getrow(0)

        # calculate target vector for query based on fasttext vectors of words and weighted tf-idf sum
        dim = int(getattr(self.model, "vector_size"))
        vec = np.zeros(dim, dtype=np.float32)
        weight_sum = 0.0

        for feature_idx, weight in zip(row.indices, row.data):
            token = str(self.feature_names[feature_idx])
            token_vec = self._get_token_vector(token)
            if token_vec is None:
                continue
            vec += float(weight) * token_vec
            weight_sum += float(weight)

        if weight_sum > 0:
            vec /= weight_sum

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm

        # if total vector is empty then return None
        if not np.any(vec):
            return None
        return vec

    def get_scores(self, query_tokens: list[str]) -> dict[int, float]:
        """
        Calculate score for documents by given query
        :param query_tokens: list of target tokens
        :return: doc_id -> score mapping
        """
        # if index is empty
        if self.doc_vectors is None:
            return {}

        # build query vector
        q_vec = self._build_query_vector(query_tokens)
        if q_vec is None:
            return {}

        # dot production to get cosine similarity
        scores = self.doc_vectors @ q_vec

        # filter scores by min_score if defined
        if self.min_score is not None:
            keep = scores > self.min_score
            doc_ids = np.nonzero(keep)[0]
        else:
            doc_ids = np.arange(scores.shape[0])

        # return scores by doc
        return {int(doc_id): float(scores[doc_id]) for doc_id in doc_ids}

    def save(self, out_dir: Path) -> None:
        """
        Save index artifacts for word2vec
        :param out_dir: directory where index artifacts will be saved.
        """
        # check if dir exists and create it
        out_dir.mkdir(parents=True, exist_ok=True)

        # raise ValueError if some core vars of index are not defined
        if self.doc_vectors is None or self.tfidf_vectorizer is None:
            raise ValueError("index is not built yet")

        # save npy with vectors
        np.save(out_dir / "doc_vectors.npy", self.doc_vectors)

        # save trained tfidf vectorizer in binary file
        with open(out_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

        # save metadata with params of indexer
        dim = int(self.doc_vectors.shape[1]) if self.doc_vectors.ndim == 2 else 0
        meta = {
            "index_type": "word2vec",
            "vector_dim": dim,
            "tfidf_max_features": self.tfidf_max_features,
            "min_score": self.min_score,
        }

        # save meta json
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: Path, model: Any, **kwargs) -> "Word2VecIndexer":
        """
        Load index artifacts for fasttext
        :param in_dir: directory with index artifacts
        :param model: word2vec-like embedding model (used for query vectors)
        :returns: loaded index instance
        """
        # check if meta information for right loading exists
        meta_path = in_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing file: {meta_path}")

        # load metadata if exists
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        tfidf_max_features = meta.get("tfidf_max_features", 50_000)
        min_score = meta.get("min_score", 0.0)

        # create instance of class
        instance = cls(model=model, tfidf_max_features=tfidf_max_features, min_score=min_score)

        # load vectors
        vectors_path = in_dir / "doc_vectors.npy"
        if not vectors_path.exists():
            raise FileNotFoundError(f"missing file: {vectors_path}")
        instance.doc_vectors = np.load(vectors_path).astype(np.float32, copy=False)

        # load trained tf-idf vectorizer
        with open(in_dir / "tfidf_vectorizer.pkl", "rb") as f:
            instance.tfidf_vectorizer = pickle.load(f)

        # check if tf-idf is actually trained
        if instance.tfidf_vectorizer is None:
            raise ValueError("tfidf vectorizer is empty after loading")

        instance.feature_names = np.asarray(instance.tfidf_vectorizer.get_feature_names_out())
        return instance
