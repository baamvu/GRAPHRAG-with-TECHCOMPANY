import sys
import io
import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from NodeRAG import NodeSearch, NodeConfig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MIMO_API_KEY = "sk-s2qd0ho978isce9y7copngi2prjtv387sursezg2q1y7851c"
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"

mimo_client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str):
    return embedding_model.encode(text).tolist()


def build_flat_rag(dataset_folder: str, collection_name: str = "tech_company"):
    print("[Flat RAG] Building ChromaDB index...")
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    for filename in sorted(os.listdir(dataset_folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(dataset_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = []
            lines = content.split("\n")
            current_chunk = ""
            for line in lines:
                if len(current_chunk) + len(line) > 1000:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    current_chunk += "\n" + line
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            for j, chunk in enumerate(chunks):
                doc_id = f"{filename}_chunk_{j}"
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk": j})
                ids.append(doc_id)

    embeddings = []
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        for text in batch:
            emb = get_embedding(text)
            embeddings.append(emb)
        print(f"  Embedded {min(i + batch_size, len(documents))}/{len(documents)} chunks")

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[Flat RAG] Saved {len(documents)} chunks to ChromaDB\n")
    return collection


def flat_rag_query(collection, query: str, n_results: int = 5) -> str:
    query_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    context = "\n\n".join(results["documents"][0])

    response = mimo_client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {"role": "system", "content": "Answer the question based on the provided context. If the context doesn't contain enough information, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0,
        max_tokens=1000
    )
    return response.choices[0].message.content


def graph_rag_query(searcher: NodeSearch, query: str) -> str:
    try:
        result = searcher.answer(query)
        return result
    except Exception as e:
        return f"Error: {e}"


def evaluate():
    print("=" * 60)
    print("EVALUATION: Flat RAG vs GraphRAG (Step 4)")
    print("=" * 60)

    dataset_folder = "C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY\\dataset"

    collection = build_flat_rag(dataset_folder)

    print("[GraphRAG] Loading NodeRAG...")
    config = NodeConfig.from_main_folder(
        "C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY"
    )
    searcher = NodeSearch(config)
    searcher.load_graph()
    searcher.load_hnsw()
    print("[GraphRAG] Ready!\n")

    test_questions = [
        "What are the main factors driving electric vehicle adoption in the US?",
        "How has Tesla's market share in the EV market changed from 2023 to 2024?",
        "What role do government policies play in promoting electric vehicles?",
        "What are the environmental benefits and concerns about electric vehicle batteries?",
        "How does charging infrastructure availability affect EV adoption rates?",
    ]

    print("Running evaluation on 5 complex questions:\n")
    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}")

        flat_answer = flat_rag_query(collection, question)
        print(f"\n[Flat RAG]:\n{flat_answer}")

        graph_answer = graph_rag_query(searcher, question)
        print(f"\n[GraphRAG]:\n{graph_answer}")

        results.append({
            "question": question,
            "flat_rag": flat_answer,
            "graph_rag": graph_answer
        })
        print()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    eval_prompt = "Compare these Flat RAG vs GraphRAG answers and identify cases where Flat RAG may hallucinate but GraphRAG answers correctly:\n\n"
    for i, r in enumerate(results, 1):
        eval_prompt += f"Question {i}: {r['question']}\n"
        eval_prompt += f"Flat RAG: {r['flat_rag'][:500]}\n"
        eval_prompt += f"GraphRAG: {r['graph_rag'][:500]}\n\n"

    eval_prompt += "Provide a summary comparison highlighting:"
    eval_prompt += "\n1. Cases where Flat RAG hallucinated or provided inaccurate information"
    eval_prompt += "\n2. Cases where GraphRAG provided more accurate/connected information"
    eval_prompt += "\n3. Overall assessment of which approach works better for this dataset"

    response = mimo_client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {"role": "system", "content": "You are an expert evaluator comparing RAG systems. Provide objective analysis."},
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.0,
        max_tokens=2000
    )

    print(response.choices[0].message.content)

    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS: Flat RAG vs GraphRAG\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"Question: {r['question']}\n")
            f.write(f"Flat RAG:\n{r['flat_rag']}\n")
            f.write(f"GraphRAG:\n{r['graph_rag']}\n")
            f.write("-" * 40 + "\n\n")
        f.write("\nANALYSIS:\n")
        f.write(response.choices[0].message.content)

    print("\nResults saved to: evaluation_results.txt")

if __name__ == "__main__":
    evaluate()
