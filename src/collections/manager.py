from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .collection import Collection


class CollectionManager:
    """Manager for collection and json"""

    def __init__(self, root_path: Path, collections_dir_name: str = "collections") -> None:
        """
        :param root_path: project root path
        :param collections_dir_name: target directory name for json files
        """
        self.root_path = Path(root_path)
        self.collections_path = self.root_path / collections_dir_name
        self.collections_path.mkdir(parents=True, exist_ok=True)

        self._collections: dict[str, Collection] = {}
        self.load_all()

    def create_collection(
        self,
        title: str,
        description: str = "",
        keywords: Optional[list[str]] = None,
        collection_id: Optional[str] = None,
        autosave: bool = True,
    ) -> Collection:
        """
        Create and store a new collection
        :param title: collection title
        :param description: collection description
        :param keywords: optional list of keywords
        :param collection_id: optional custom id
        :param autosave: save json right after creation
        :returns: created collection object
        """
        normalised_title = str(title).strip()
        if not normalised_title:
            raise ValueError("collection title is required")

        new_id = str(collection_id).strip() if collection_id else str(uuid4())
        if new_id in self._collections:
            raise ValueError(f'collection "{new_id}" already exists')

        normalized_keywords = []
        for keyword in keywords or []:
            value = str(keyword).strip()
            if value and value not in normalized_keywords:
                normalized_keywords.append(value)

        collection = Collection(
            id=new_id,
            title=normalised_title,
            description=str(description).strip(),
            keywords=normalized_keywords,
        )
        self._collections[new_id] = collection

        if autosave:
            self.save_collection(new_id)
        return collection

    def get_collection(self, collection_id: str) -> Collection:
        """
        Return collection by id or raise error
        :param collection_id: collection id
        :returns: collection object
        """
        if collection_id not in self._collections:
            raise KeyError(f'collection "{collection_id}" not found')
        return self._collections[str(collection_id).strip()]

    def list_collections(self) -> list[Collection]:
        """
        Return all collections
        :returns: list of collections
        """
        return list(self._collections.values())

    def delete_collection(self, collection_id: str, remove_file: bool = True) -> bool:
        """
        Delete collection from memory and optional file
        :param collection_id: collection id
        :param remove_file: remove json file if exists
        :returns: true if removed
        """
        normalized_id = str(collection_id).strip()
        if normalized_id not in self._collections:
            return False

        del self._collections[normalized_id]

        if remove_file:
            file_path = self._collection_file_path(normalized_id)
            if file_path.exists():
                file_path.unlink()
        return True

    def add_paper(self, collection_id: str, paper_id: str, autosave: bool = True) -> bool:
        """
        Add paper to collection
        :param collection_id: collection id
        :param paper_id: paper id
        :param autosave: save json after update
        :returns: true if added
        """
        collection = self.get_collection(collection_id)
        changed = collection.add_paper(paper_id)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def remove_paper(self, collection_id: str, paper_id: str, autosave: bool = True) -> bool:
        """
        Remove paper from collection
        :param collection_id: collection id
        :param paper_id: paper id
        :param autosave: save json after update
        :returns: true if removed
        """
        collection = self.get_collection(collection_id)
        changed = collection.remove_paper(paper_id)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def add_keyword(self, collection_id: str, keyword: str, autosave: bool = True) -> bool:
        """
        Add keyword to collection
        :param collection_id: collection id
        :param keyword: keyword text
        :param autosave: save json after update
        :returns: true if added
        """
        collection = self.get_collection(collection_id)
        changed = collection.add_keyword(keyword)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def remove_keyword(self, collection_id: str, keyword: str, autosave: bool = True) -> bool:
        """
        Remove keyword from collection
        :param collection_id: collection id
        :param keyword: keyword text
        :param autosave: save json after update
        :returns: true if removed
        """
        collection = self.get_collection(collection_id)
        changed = collection.remove_keyword(keyword)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def change_title(self, collection_id: str, new_title: str, autosave: bool = True) -> bool:
        """
        Update collection title
        :param collection_id: collection id
        :param new_title: new title
        :param autosave: save json after update
        :returns: true if changed
        """
        collection = self.get_collection(collection_id)
        changed = collection.change_title(new_title)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def change_description(self, collection_id: str, new_description: str, autosave: bool = True) -> bool:
        """
        Update collection description
        :param collection_id: collection id
        :param new_description: new description
        :param autosave: save json after update
        :returns: true if changed
        """
        collection = self.get_collection(collection_id)
        changed = collection.change_description(new_description)
        if changed and autosave:
            self.save_collection(collection.id)
        return changed

    def save_collection(self, collection_id: str) -> Path:
        """
        Save one collection to json file
        :param collection_id: collection id
        :returns: path to saved file
        """
        collection = self.get_collection(collection_id)
        file_path = self._collection_file_path(collection.id)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(collection.to_dict(), f, ensure_ascii=False, indent=2)
        return file_path

    def save_all_to_json(self) -> None:
        """Save all loaded collections to json files"""
        for collection in self._collections.values():
            self.save_collection(collection.id)

    def load_collection(self, collection_id: str) -> Collection:
        """
        Load one collection from json file
        :param collection_id: collection id
        :returns: loaded collection object
        """
        file_path = self._collection_file_path(collection_id)
        if not file_path.exists():
            raise FileNotFoundError(f"missing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        collection = Collection.from_dict(payload)
        self._collections[collection.id] = collection
        return collection

    def load_all(self) -> None:
        """Load all collections from json directory"""
        self._collections.clear()
        for file_path in sorted(self.collections_path.glob("*.json")):
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            collection = Collection.from_dict(payload)
            self._collections[collection.id] = collection

    def get_profile_text(self, collection_id: str) -> str:
        """
        Build collection profile text for recommendation logic
        :param collection_id: collection id
        :returns: text from title, description and keywords
        """
        collection = self.get_collection(collection_id)

        parts = [collection.title.strip(), collection.description.strip(), " ".join(collection.keywords).strip()]
        return " ".join([part for part in parts if part]).strip()

    def to_list_of_dicts(self) -> list[dict]:
        """
        Dump all collections to list of dict
        :returns: list with serialized collections
        """
        return [collection.to_dict() for collection in self.list_collections()]

    def _collection_file_path(self, collection_id: str) -> Path:
        """
        Build json file path for collection
        :param collection_id: collection id
        :returns: path to json
        """
        normalized_id = str(collection_id).strip()
        if not normalized_id:
            raise ValueError("collection id is required")
        return self.collections_path / f"{normalized_id}.json"
