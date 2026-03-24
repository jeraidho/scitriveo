from __future__ import annotations
from time import perf_counter
from typing import Any, Optional
import pandas as pd
from src.collections.manager import CollectionManager
from src.services.search_service import SearchService


class RecommendationService:
    """Service for collection recommendations with profile and studies (seed) fusion"""

    def __init__(
        self,
        collection_manager: CollectionManager,
        search_service: SearchService,
        docs_df: pd.DataFrame,
        default_index_name: str = "fasttext",
        seed_text_column: str = "index_text_lemmatised",
        rrf_k: int = 60,
    ) -> None:
        """
        :param collection_manager: manager with collections state
        :param search_service: search service with metadata enrichment
        :param docs_df: dataframe with corpus metadata
        :param default_index_name: default semantic index for recommendation
        :param seed_text_column: target text column for per-seed queries
        :param rrf_k: RRF denominator constant
        """
        self.collection_manager = collection_manager
        self.search_service = search_service
        self.docs_df = docs_df.reset_index(drop=True)
        self.default_index_name = default_index_name
        self.seed_text_column = seed_text_column
        self.rrf_k = rrf_k

        # map paper id -> dataframe row for fast access
        self._docs_by_paper_id = self.docs_df.set_index("id", drop=False) if "id" in self.docs_df.columns else None

    def recommend(
        self,
        collection_id: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        profile_candidates: int = 100,
        seed_candidates: int = 30,
        max_seeds: int = 10,
    ) -> dict[str, Any]:
        """
        Main function to build recommendations for user's collection
        :param collection_id: target collection id
        :param top_k: number of recommendations
        :param filters: optional search filters
        :param profile_candidates: candidate size for profile-based search
        :param seed_candidates: candidate size for each research (seed) search
        :param max_seeds: limit number of seeds to search
        :returns: recommendation response
        """
        # recommendation pipeline:
        # 1 profile-based search by collection text (title+description+keywords)
        # 2 research search (studies as seeds) for each added paper
        # 3 RRF fusion:
        # internal seed RRF across seed lists
        # final RRF between profile list and fused seed list

        start = perf_counter()
        collection = self.collection_manager.get_collection(collection_id)
        if collection is None:
            raise KeyError(f'collection "{collection_id}" not found')

        # get stats from collection
        excluded_ids = set(collection.added_papers)
        profile_text = self.collection_manager.get_profile_text(collection_id)

        # profile-based ranking
        profile_ranking: list[dict[str, Any]] = []
        if profile_text:
            profile_response = self.search_service.search_with_filters(
                query=profile_text,
                index_name=self.default_index_name,
                top_k=profile_candidates,
                filters=filters,
            )
            profile_ranking = profile_response["results"]

        # seed-based ranking (study-based search + internal seed fusion)
        seed_ids = list(collection.added_papers)[:max_seeds]
        # now we need to get lists of ranks for RRF
        # shape: list[list[dict[str, ...]]]
        seed_rankings = []
        for seed_id in seed_ids:
            seed_query = self._build_seed_query(seed_id)
            if not seed_query:
                continue
            seed_response = self.search_service.search_with_filters(
                query=seed_query,
                index_name=self.default_index_name,
                top_k=seed_candidates,
                filters=filters,
            )
            if seed_response["results"]:
                seed_rankings.append(seed_response["results"])

        fused_seed_ranking = self._fuse_rrf(seed_rankings, self.rrf_k)

        # final fusion between profile list and seed list
        # the same type as seed_rankings
        final_rankings = []
        if profile_ranking:
            final_rankings.append(profile_ranking)
        if fused_seed_ranking:
            final_rankings.append(fused_seed_ranking)

        fused_final = self._fuse_rrf(final_rankings, self.rrf_k)

        # remove added papers and trim
        recommendations = []
        for item in fused_final:
            paper_id = item["paper_id"]
            if not paper_id or paper_id in excluded_ids:
                continue
            recommendations.append(item)
            if len(recommendations) >= top_k:
                break

        elapsed_ms = (perf_counter() - start) * 1000.0

        # add metatag for information about type of recommendation
        mode = "warm_start" if collection.added_papers else "cold_start"

        return {
            "collection_id": collection.id,
            "mode": mode,
            "index_name": self.default_index_name,
            "top_k": top_k,
            "results_count": len(recommendations),
            "elapsed_ms": round(elapsed_ms, 3),
            "results": recommendations,
        }

    def _build_seed_query(self, paper_id: str) -> str:
        """
        Build seed query text from paper metadata
        :param paper_id: corpus paper id
        :returns: query string
        """
        if self._docs_by_paper_id is None:
            return ""
        if paper_id not in self._docs_by_paper_id.index:
            return ""

        row = self._docs_by_paper_id.loc[paper_id]

        # prefer preprocessed text to match index representation
        if self.seed_text_column in row.index:
            prepared_text = str(row[self.seed_text_column] or "").strip()
            if prepared_text:
                return prepared_text

        # fallback to raw text if prepared text is missing
        title = str(row["title"] or "").strip() if "title" in row.index else ""
        abstract = str(row["abstract"] or "").strip() if "abstract" in row.index else ""
        return " ".join([p for p in [title, abstract] if p]).strip()

    @staticmethod
    def _fuse_rrf(rankings: list[list[dict[str, Any]]], rrf_k: int) -> list[dict[str, Any]]:
        """
        Fuse many rankings using RRF
        :param rankings: list of rankings (each ranking is list of hit dicts from search)
        :param rrf_k: denominator constant in RRF formula
        :returns: fused list sorted by recommendation_score
        """
        score_map: dict[str, float] = {}
        best_item_map: dict[str, dict[str, Any]] = {}

        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                paper_id = item.get("paper_id")
                if not paper_id:
                    continue

                # RRF score contribution: 1 / (k + rank)
                score_map[paper_id] = score_map.get(paper_id, 0.0) + (1.0 / float(rrf_k + rank))

                # keep first occurrence as template
                if paper_id not in best_item_map:
                    best_item_map[paper_id] = dict(item)

        fused = []
        for paper_id, score in score_map.items():
            payload = dict(best_item_map[paper_id])
            payload["recommendation_score"] = round(float(score), 8)
            fused.append(payload)

        fused.sort(key=lambda x: x["recommendation_score"], reverse=True)
        return fused

