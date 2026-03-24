class SearchEngine:
    def __init__(self, indexer, preprocessor):
        """
        Initialisation of search engine
        :param indexer: class of indexer to search with
        :param preprocessor: class of text preprocessor to preprocess queries
        """
        # search engine takes an indexer and query preprocessor
        # define indexer from /src/indexers
        self.indexer = indexer

        # preprocessor (our main is in /src/preprocessing.py)
        self.preprocessor = preprocessor

        # var for storing text without preprocessing
        self.original_texts, self.processed_texts = [], []

    def fit(self, original_texts: list[str], processed_texts: list[str]) -> None:
        """
        Function to add text and create index based on it
        :param original_texts: original texts from dataset
        :param processed_texts: preprocessed texts
        :return: None
        """
        # store original texts without preprocessing
        self.original_texts, self.processed_texts = original_texts, processed_texts

        # build index from already preprocessed data
        self.indexer.build_index(processed_texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Main function to search string based
        :param query: str text to search
        :param top_k: number of documents by score to return
        :return: list of relevant documents by score
        """
        # preprocess query to match index terms
        query_tokens = self.preprocessor.process_string(query).split()

        if not query_tokens:
            return []

        # get scores from indexer
        scores = self.indexer.get_scores(query_tokens)

        # rank documents by score in descending sort
        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        # for each top_k doc save it to results with score and original text
        for doc_id, score in ranked_ids[:top_k]:
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "text": self.original_texts[doc_id],
                "preprocessed_text": self.processed_texts[doc_id]
            })

        return results
