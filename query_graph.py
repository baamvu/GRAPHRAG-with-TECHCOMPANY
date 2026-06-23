import sys
import io
from NodeRAG import NodeSearch, NodeConfig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("GRAPH RAG - QUERY (Step 3)")
    print("=" * 60)

    config = NodeConfig.from_main_folder(
        "C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY"
    )

    print("\n[1/3] Loading graph and indices...")
    searcher = NodeSearch(config)
    searcher.load_graph()

    print("[2/3] Loading HNSW index...")
    searcher.load_hnsw()

    print("[3/3] System ready!\n")

    test_queries = [
        "What are the main electric vehicle manufacturers in the US?",
        "How has Tesla's market share changed over time?",
        "What government policies support EV adoption?",
        "What are the environmental benefits of electric vehicles?",
        "How does charging infrastructure affect EV adoption?",
    ]

    print("Running sample queries:\n")
    for i, query in enumerate(test_queries, 1):
        print(f"--- Question {i}: {query}")
        try:
            result = searcher.answer(query)
            print(f"Answer: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nYou can enter your own questions (type 'quit' to exit):")
    while True:
        user_query = input("\nQuestion: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if not user_query:
            continue
        try:
            result = searcher.answer(user_query)
            print(f"Answer: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
