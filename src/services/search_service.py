from __future__ import annotations
from time import perf_counter
from typing import Any, Optional
import pandas as pd
from src.search.factory import SearchEngineFactory


class SearchService:
    """Service to run search and enrich results with metadata and do filtered search"""

    DEFAULT_METADATA_COLUMNS = (
        "id",
        "doi",
        "author",
        "title",
        "publication_year",
        "journal",
        "is_retracted",
        "cited_by_count",
        "field",
        "source",
    )

    def __init__(
            self,
            engine_factory: SearchEngineFactory,
            docs_df: pd.DataFrame,
            metadata_columns: Optional[list[str]] = None,
            original_text_column: str = "abstract",
            processed_text_column: str = "index_text_lemmatised",
    ) -> None:
        """
        :param engine_factory: factory with ready indexers
        :param docs_df: dataframe with metadata and text columns
        :param metadata_columns: metadata columns to return in each hit
        :param original_text_column: column with original text
        :param processed_text_column: column with preprocessed text
        """
        self.engine_factory = engine_factory
        self.docs_df = docs_df.reset_index(drop=True)
        self.original_text_column = original_text_column
        self.processed_text_column = processed_text_column

        if metadata_columns is None:
            metadata_columns = [c for c in self.DEFAULT_METADATA_COLUMNS if c in self.docs_df.columns]
        self.metadata_columns = metadata_columns

        self._validate_dataframe_columns()

    def _validate_dataframe_columns(self) -> None:
        """
        Check required columns in dataframe
        """
        required_columns = [self.original_text_column, self.processed_text_column]
        missing = [c for c in required_columns if c not in self.docs_df.columns]
        if missing:
            raise ValueError(f"missing required columns in docs_df: {missing}")

    def search(self, query: str, index_name: str, top_k: int = 10) -> dict[str, Any]:
        """
        Run search and enrich hits with dataframe metadata
        :param query: query string
        :param index_name: target indexer name
        :param top_k: number of docs to return
        :returns: response with search metadata and hits
        """
        return self.search_with_filters(
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=None,
        )

    def search_with_filters(
            self,
            query: str,
            index_name: str,
            top_k: int = 10,
            filters: Optional[dict[str, Any]] = None,
            oversample_factor: int = 5,
    ) -> dict[str, Any]:
        """
        Run search, enrich hits with metadata and apply filters
        Supported filter keys:
        - is_retracted
        - publication_year_from
        - publication_year_to
        - journal_contains
        - field_in
        - cited_by_count_min
        - cited_by_count_max
        :param query: query string
        :param index_name: target indexer name
        :param top_k: number of docs to return
        :param filters: optional metadata filters
        :param oversample_factor: multiplier for candidate retrieval before filtering
        :returns: response with search metadata and hits
        """
        start = perf_counter()

        engine = self.engine_factory.get_engine(index_name=index_name)

        # get more candidates if filters are enabled
        candidate_k = top_k
        if filters:
            max_docs = len(self.docs_df)
            candidate_k = min(max_docs, max(top_k, top_k * max(1, oversample_factor)))

        raw_hits = engine.search(query=query, top_k=candidate_k)

        elapsed_ms = (perf_counter() - start) * 1000.0

        hits = []
        for hit in raw_hits:
            enriched = self._enrich_hit(hit)
            if self._passes_filters(enriched, filters):
                hits.append(enriched)
            if len(hits) >= top_k:
                break

        return {
            "query": query,
            "index_name": index_name,
            "top_k": top_k,
            "filters": filters or {},
            "results_count": len(hits),
            "elapsed_ms": round(elapsed_ms, 3),
            "results": hits,
        }

    def _enrich_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich one search hit with metadata from docs df
        :param hit: raw hit from SearchEngine
        :returns: enriched hit dict
        """
        doc_id = int(hit["id"])
        row = self.docs_df.iloc[doc_id]
        paper_id = row["id"] if "id" in row.index else None

        enriched: dict[str, Any] = {
            "paper_id": None if pd.isna(paper_id) else str(paper_id),
            "doc_id": doc_id,
            "score": hit["score"],
            "text": hit["text"],
            "preprocessed_text": hit["preprocessed_text"],
        }

        for column in self.metadata_columns:
            value = row[column] if column in row.index else None
            if pd.isna(value):
                value = None
            enriched[column] = value

        return enriched

    @staticmethod
    def _passes_filters(hit: dict[str, Any], filters: Optional[dict[str, Any]]) -> bool:
        """
        Check if enriched hit passes metadata filters
        :param hit: enriched hit dict
        :param filters: filters dict
        :returns: true if hit should be kept
        """
        if not filters:
            return True

        # filter by retraction flag
        if "is_retracted" in filters:
            expected = bool(filters["is_retracted"])
            actual = bool(hit["is_retracted"]) if hit["is_retracted"] is not None else False
            if actual != expected:
                return False

        # filter by publication year interval
        year = hit["publication_year"]
        if "publication_year_from" in filters and year is not None:
            if int(year) < int(filters["publication_year_from"]):
                return False
        if "publication_year_to" in filters and year is not None:
            if int(year) > int(filters["publication_year_to"]):
                return False

        # filter by journal substring
        if "journal_contains" in filters:
            needle = str(filters["journal_contains"]).strip().lower()
            journal = str(hit["journal"] or "").lower()
            if needle and needle not in journal:
                return False

        # filter by field inclusion
        if "field_in" in filters:
            allowed = {str(v).strip().lower() for v in filters["field_in"] if str(v).strip()}
            field = str(hit["field"] or "").strip().lower()
            if allowed and field not in allowed:
                return False

        # filter by cited count interval
        cited = hit["cited_by_count"]
        if cited is not None:
            if "cited_by_count_min" in filters and int(cited) < int(filters["cited_by_count_min"]):
                return False
            if "cited_by_count_max" in filters and int(cited) > int(filters["cited_by_count_max"]):
                return False

        return True
