from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from src.collections.manager import CollectionManager
from src.configs.app_config import AppConfig, AppPaths
from src.data.preprocessing import TextPreprocessor
from src.indexers.word2vec_indexer import W2VWrapper
from src.search.factory import SearchEngineFactory
from src.services import IndexBuildService, RecommendationService, SearchService
from src.services.rag_service_ollama import RAGServiceOllama
import torch


class AppContainer:
    """Application composition root for services and managers"""

    def __init__(
            self,
            config: AppConfig,
            preprocessor: Optional[Any] = None,
            models: Optional[dict[str, Any]] = None,
            docs_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        :param config: app config with paths and startup settings
        :param preprocessor: query preprocessor instance
        :param models: optional preloaded models dict
        :param docs_df: optional preloaded corpus dataframe
        """
        self.config = config
        self.paths = config.paths

        self._ensure_base_directories()

        # init basic dependencies
        # define preprocessor if not given user's one
        self.preprocessor = preprocessor or TextPreprocessor()

        # define docs dataframe if user's given
        self.docs_df = docs_df if docs_df is not None else self._load_docs_df()

        # load basic models from config if not models
        self.models = models or self._load_models()

        # init collection manager with persistence in root/collections
        self.collection_manager = CollectionManager(
            root_path=self.paths.root,
            collections_dir_name=self.paths.collections_dir.name,
        )

        # init index builder
        self.index_build_service = IndexBuildService(
            corpus_csv_path=self.paths.corpus_preprocessed_csv_path,
            indexes_root=self.paths.indexes_dir,
            processed_text_column=self.config.processed_text_column,
            models=self.models,
        )

        # ensure that indexes are created
        if self.config.ensure_indexes_on_start:
            self.index_build_service.ensure_indexes(
                index_names=list(self.config.indexes_to_ensure),
                rebuild=self.config.rebuild_indexes_on_start,
            )

        # init search factory
        self.search_engine_factory = SearchEngineFactory(
            indexes_root=self.paths.indexes_dir,
            preprocessor=self.preprocessor,
            original_texts=self.docs_df[self.config.original_text_column].fillna("").astype(str).tolist(),
            processed_texts=self.docs_df[self.config.processed_text_column].fillna("").astype(str).tolist(),
            models=self.models,
        )

        # init search service
        self.search_service = SearchService(
            engine_factory=self.search_engine_factory,
            docs_df=self.docs_df,
            original_text_column=self.config.original_text_column,
            processed_text_column=self.config.processed_text_column,
        )

        # init recommendation service
        self.recommendation_service = RecommendationService(
            collection_manager=self.collection_manager,
            search_service=self.search_service,
            docs_df=self.docs_df,
            default_index_name=self.config.recommendation_index_name,
            seed_text_column=self.config.seed_text_column,
            rrf_k=self.config.recommendation_rrf_k,
        )

        # service for rag
        # define device based on available backend
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # initialise rag service
        # self.rag_service = RAGService(
        #     model_path=(self.paths.models_dir / "clara" / "compression-16"),
        #     device=device)
        self.rag_service = RAGServiceOllama(model_name="mistral")


    @classmethod
    def from_root(
            cls,
            root_path: Path,
            preprocessor: Optional[Any] = None,
            models: Optional[dict[str, Any]] = None,
            docs_df: Optional[pd.DataFrame] = None,
            **config_kwargs: Any,
    ) -> "AppContainer":
        """
        Build container from root directory and optional config overrides
        :param root_path: path to project root
        :param preprocessor: optional preprocessor object
        :param models: optional preloaded models
        :param docs_df: optional preloaded dataframe
        :param config_kwargs: optional AppConfig overrides
        :returns: app container instance
        """
        paths = AppPaths(root=Path(root_path))
        config = AppConfig(paths=paths, **config_kwargs)
        return cls(config=config, preprocessor=preprocessor, models=models, docs_df=docs_df)

    def search(
            self,
            query: str,
            index_name: str,
            top_k: int = 10,
            filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Facade method for search operations
        :param query: query text
        :param index_name: bm25, word2vec or fasttext
        :param top_k: max results
        :param filters: optional metadata filters
        :returns: search response
        """
        return self.search_service.search_with_filters(
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters,
        )

    def create_collection(
            self,
            title: str,
            description: str = "",
            keywords: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Facade method to create collection
        :param title: collection title
        :param description: collection description
        :param keywords: optional keywords
        :returns: serialised collection
        """
        collection = self.collection_manager.create_collection(
            title=title,
            description=description,
            keywords=keywords,
            autosave=True,
        )
        return collection.to_dict()

    def add_paper_to_collection(self, collection_id: str, paper_id: str) -> bool:
        """
        Facade method to add paper to collection.
        :param collection_id: collection id
        :param paper_id: stable corpus paper id
        :returns: true if added
        """
        return self.collection_manager.add_paper(collection_id=collection_id, paper_id=paper_id, autosave=True)

    def recommend(
            self,
            collection_id: str,
            top_k: int = 10,
            filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Facade method for recommendation service call
        :param collection_id: collection id
        :param top_k: max recommendations
        :param filters: optional metadata filters
        :returns: recommendation response
        """
        return self.recommendation_service.recommend(
            collection_id=collection_id,
            top_k=top_k,
            filters=filters,
        )

    def _ensure_base_directories(self) -> None:
        """
        Internal function to ensure the default directories from appconfig
        :return: None
        """
        self.paths.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.paths.collections_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.csv_dir.mkdir(parents=True, exist_ok=True)

    def _load_docs_df(self) -> pd.DataFrame:
        """
        Load dataframe with corpus metadata and text.
        :returns: dataframe
        """
        path = self.paths.corpus_preprocessed_csv_path
        if not path.exists():
            # fallback for current project layout
            fallback = self.paths.src_dir / "data" / "abstracts_lemmatized.csv"
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(
                    f"preprocessed csv not found. checked: {self.paths.corpus_preprocessed_csv_path} and {fallback}"
                )

        df = pd.read_csv(path).reset_index(drop=True)
        if self.config.original_text_column not in df.columns:
            raise ValueError(f'missing column "{self.config.original_text_column}" in corpus csv')
        if self.config.processed_text_column not in df.columns:
            raise ValueError(f'missing column "{self.config.processed_text_column}" in corpus csv')
        return df

    def _load_models(self) -> dict[str, Any]:
        """
        Load word2vec and fasttext models if files are present.
        Model loading is best-effort:
        - word2vec from .vec via KeyedVectors + W2VWrapper
        - fasttext from facebook .bin via gensim loader

        :returns: models dict
        """
        models: dict[str, Any] = {}

        # load word2vec vectors
        w2v_path = self.paths.word2vec_model_path
        if not w2v_path.exists():
            fallback_w2v = self.paths.src_dir / "data" / "wiki-news-300d-1M.vec"
            w2v_path = fallback_w2v if fallback_w2v.exists() else w2v_path

        if w2v_path.exists():
            from gensim.models import KeyedVectors

            kv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=False)
            models["word2vec"] = W2VWrapper(kv)

        # load fasttext model
        ft_path = self.paths.fasttext_model_path
        if not ft_path.exists():
            fallback_ft = self.paths.src_dir / "data" / "model.bin"
            ft_path = fallback_ft if fallback_ft.exists() else ft_path

        if ft_path.exists():
            from gensim.models.fasttext import load_facebook_model

            models["fasttext"] = load_facebook_model(str(ft_path))

        return models

    # def ask_collection(self, collection_id: str, question: str) -> str:
    #     """
    #     Facade method to answer questions based on collection content
    #     :param collection_id: target collection uuid
    #     :param question: user inquiry
    #     :returns: generated answer from the rag model
    #     """
    #     collection = self.collection_manager.get_collection(collection_id)
    #     if not collection.added_papers:
    #         return "Collection is empty"
    #
    #     # fetch titles and abstracts for all papers in the collection
    #     mask = self.docs_df['id'].isin(collection.added_papers)
    #     subset = self.docs_df[mask]
    #
    #     # concatenate title and abstract for each document into context strings
    #     context_list = []
    #     for _, row in subset.iterrows():
    #         doc_representation = f"title: {row.get('title', '')} | abstract: {row.get('abstract', '')}"
    #         context_list.append(doc_representation)
    #
    #     # delegate generation to the rag service
    #     return self.rag_service.generate_answer(question, context_list)

    def ask_collection(self, collection_id: str, question: str) -> str:
        """
        Facade method to answer questions based on enriched collection metadata
        :param collection_id: target collection unique identifier
        :param question: the user research question
        :returns: generated answer from the model
        """
        collection = self.collection_manager.get_collection(collection_id)
        if not collection.added_papers:
            return "the selected collection contains no papers"

        # fetch metadata from the corpus dataframe for all papers in the workspace
        mask = self.docs_df['id'].isin(collection.added_papers)
        subset = self.docs_df[mask]

        # build a rich context string for each paper including metadata
        context_list = []
        for _, row in subset.iterrows():
            # format each document as a structured snippet of knowledge
            item = (
                f"title: {row.get('title', 'n/a')}\n"
                f"authors: {row.get('author', 'unknown')}\n"
                f"year: {row.get('publication_year', 'unknown')}\n"
                f"journal: {row.get('journal', 'n/a')}\n"
                f"citations: {row.get('cited_by_count', 'zero')}\n"
                f"abstract: {row.get('abstract', 'no content available')}\n"
                f"---"
            )
            context_list.append(item)

        # delegate the synthesis task to the generative service
        return self.rag_service.generate_answer(question, context_list)