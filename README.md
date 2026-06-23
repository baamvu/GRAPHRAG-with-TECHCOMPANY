# GRAPHRAG-with-TECHCOMPANY

GraphRAG system using NodeRAG to analyze tech company and electric vehicle data.

## Setup

```bash
pip install NodeRAG chromadb openai sentence-transformers
```

## Project Structure

```
GRAPHRAG-with-TECHCOMPANY/
├── dataset/              # Original data (70 documents)
├── input/                # Input data for NodeRAG
├── cache/                # Build cache and results
├── info/                 # Logs and indices
├── Node_config.yaml      # NodeRAG configuration
├── build_graph.py        # Step 1 & 2: Indexing + Build graph
├── query_graph.py        # Step 3: Query interface
├── evaluate.py           # Step 4: Compare Flat RAG vs GraphRAG
└── evaluation_results.txt # Evaluation results
```

## Usage

### Step 1 & 2: Build Graph (NodeRAG - Option C)
```bash
python build_graph.py
```
- Extracts entities and relationships from dataset
- Builds knowledge graph using NodeRAG
- Uses MIMO API for LLM, local MiniLM for embeddings

### Step 3: Query
```bash
python query_graph.py
```
- Runs sample queries
- Allows interactive question input

### Step 4: Evaluation
```bash
python evaluate.py
```
- Compares Flat RAG (ChromaDB) vs GraphRAG (NodeRAG)
- Identifies cases where Flat RAG hallucinates

## Configuration

Edit `Node_config.yaml` to change:
- `model_config.model_name`: LLM model (mimo-v2.5-pro)
- `embedding_config.embedding_model_name`: Embedding model
- `config.chunk_size`: Text chunk size
- `config.language`: Language (English)

## API Keys

- **MIMO API**: Set in `Node_config.yaml` (model_config.api_keys)
- **Embedding**: Uses local sentence-transformers (no API needed)
