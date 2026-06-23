import sys
import io
import asyncio
from NodeRAG import NodeRag, NodeConfig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("GRAPH RAG - TECH COMPANY CORPUS")
    print("Step 1 & 2: Indexing + Build Graph (NodeRAG)")
    print("=" * 60)

    config = NodeConfig.from_main_folder(
        "C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY"
    )

    print("\n[1/2] Initializing NodeRAG pipeline...")
    node_rag = NodeRag(config, web_ui=True)

    print("[2/2] Running build pipeline (extract entities, relations, build graph)...")
    print("       This process may take a few minutes...\n")

    node_rag.run()

    print("\n" + "=" * 60)
    print("BUILD COMPLETED!")
    print("Graph saved in: cache/")
    print("=" * 60)

if __name__ == "__main__":
    main()
