# SciTriveo: Intellectual Research Copilot

**SciTriveo** is an advanced Information Retrieval (IR) system designed for intellectual research.

It combines classical lexical retrieval (BM25) with modern semantic search (Word2Vec, FastText) and a dynamic recommendation engine, SciTriveo transforms creates an interactive research workspace.

## Key Features

*   **Hybrid Search Engine**: Supports three distinct indexing strategies:
    *   **BM25 (Okapi)**: High-performance lexical matching
    *   **Word2Vec**: Contextual similarity using TF-IDF weighted mean vectors
    *   **FastText**: Subword-aware semantic retrieval to handle out-of-vocabulary scientific terms
*   **Research Collections**: Create and manage persistent collections of papers.
*   **Context-Aware RecSys**: A recommendation system using **Reciprocal Rank Fusion (RRF)** to suggest new papers based on the "centroid" of your current research collection.

---

##  System Architecture

The project follows a composition root pattern. The `AppContainer` serves as the central hub, managing services, indices, and managers.

### Core Components
*   **`src/indexers`**: Implements the Strategy pattern for different retrieval models
*   **`src/services`**: Orchestrates high-level logic (e.g., `SearchService` for metadata enrichment and `RecommendationService` for RRF logic)
*   **`src/collections`**: Handles persistence and state of user workspaces
*   **`src/data`**: A linguistic pipeline for cleaning, language detection, and lemmatisation

---

## 📂 Project Structure

```text
.
├── collections/        # Persistent JSON storage for user workspaces
├── csv/                # Raw and preprocessed (lemmatised) datasets
├── docs/               # Technical API documentation (HTML)
├── indexes/            # Serialized index artifacts (Pickle, NPY)
├── models/             # Pre-trained static model weights (W2V, FastText)
├── src/                # Primary source code
│   ├── app/            # Application container
│   ├── cli/            # Command Line Interface
│   ├── collections/    # Collection collections logic
│   ├── configs/        # System-wide paths and runtime settings
│   ├── data/           # NLP preprocessing and data cleaning
│   ├── indexers/       # Search model implementations (BM25, W2V, FT)
│   ├── search/         # Search engine factory and core search logic
│   ├── services/       # High-level logic (Search, RecSys)
│   └── spellchecker/   # Scientific spelling correction
└── requirements.txt    # Project dependencies
```

---

## Installation & Setup

### Installation
1. Clone the repository and navigate to the root:
   ```bash
   git clone https://github.com/your-repo/scitriveo.git
   cd scitriveo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place model files (`wiki-news-300d-1M.vec`, `fasttext_model.bin` *from facebook fasttext vectors*) into the `models/` directory.

---

## Usage (CLI)

SciTriveo is designed to be executed as a module.

### 1. Interactive Shell
To avoid the overhead of loading heavy embedding models for every query, use the interactive mode:
```bash
python -m src.cli interactive
```
**Inside the shell:**
```text
scitriveo > search "morphology in caucasian languages" --index fasttext --top-k 3
scitriveo > collection create --title "Linguistics" --desc "Focus on morphology"
scitriveo > recommend --id <COLLECTION_UUID>
```

### 2. Standard CLI Commands
Perform one-off operations directly from the terminal:
*   **Search**:
    ```bash
    python -m src.cli search "deep learning" --index bm25
    ```
*   **List Collections**:
    ```bash
    python -m src.cli collection list --show-papers
    ```
    
