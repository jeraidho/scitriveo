from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    """Create utc timestamp in iso format"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Collection:
    """Data object for one user collection"""

    id: str
    title: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    added_papers: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def add_paper(self, paper_id: str) -> bool:
        """
        Add paper id if it does not exist
        :param paper_id: paper id from corpus
        :returns: true if paper was added
        """
        normalised_id = str(paper_id).strip()
        if not normalised_id:
            return False
        if normalised_id in self.added_papers:
            return False
        self.added_papers.append(normalised_id)
        self._touch()
        return True

    def remove_paper(self, paper_id: str) -> bool:
        """
        Remove paper id if it exists
        :param paper_id: paper id from corpus
        :returns: true if paper was removed
        """
        normalized_id = str(paper_id).strip()
        if normalized_id not in self.added_papers:
            return False
        self.added_papers.remove(normalized_id)
        self._touch()
        return True

    def add_keyword(self, keyword: str) -> bool:
        """
        Add keyword if it does not exist
        :param keyword: keyword text
        :returns: true if keyword was added
        """
        normalized_keyword = str(keyword).strip()
        if not normalized_keyword:
            return False
        if normalized_keyword in self.keywords:
            return False
        self.keywords.append(normalized_keyword)
        self._touch()
        return True

    def remove_keyword(self, keyword: str) -> bool:
        """
        Remove keyword if it exists
        :param keyword: keyword text
        :returns: true if keyword was removed
        """
        normalized_keyword = str(keyword).strip()
        if normalized_keyword not in self.keywords:
            return False
        self.keywords.remove(normalized_keyword)
        self._touch()
        return True

    def change_title(self, new_title: str) -> bool:
        """
        Change title if value is not empty and changed
        :param new_title: new title text
        :returns: true if changed
        """
        value = str(new_title).strip()
        if not value:
            return False
        if value == self.title:
            return False
        self.title = value
        self._touch()
        return True

    def change_description(self, new_description: str) -> bool:
        """
        Change description
        :param new_description: new description text
        :returns: true if changed
        """
        value = str(new_description).strip()
        if value == self.description:
            return False
        self.description = value
        self._touch()
        return True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert collection object to dict
        :returns: serializable dict
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "keywords": list(self.keywords),
            "added_papers": list(self.added_papers),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Collection":
        """
        Build collection from dict
        :param payload: dict payload
        :returns: collection object
        """
        collection_id = str(payload.get("id", "")).strip()
        title = str(payload.get("title", "")).strip()
        if not collection_id:
            raise ValueError("collection id is required")
        if not title:
            raise ValueError("collection title is required")

        keywords = [str(k).strip() for k in payload.get("keywords", []) if str(k).strip()]
        added_papers = [str(p).strip() for p in payload.get("added_papers", []) if str(p).strip()]

        return cls(
            id=collection_id,
            title=title,
            description=str(payload.get("description", "")).strip(),
            keywords=keywords,
            added_papers=added_papers,
            created_at=str(payload.get("created_at", _utc_now_iso())),
            updated_at=str(payload.get("updated_at", _utc_now_iso())),
        )

    def _touch(self) -> None:
        """Update timestamp after collection mutation"""
        self.updated_at = _utc_now_iso()
