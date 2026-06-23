import sys
import io
import os
import json
import time
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


def build_flat_rag(dataset_folder: str, collection_name: str = "tech_company_bench"):
    print("[Flat RAG] Building ChromaDB index...")
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)

    documents, metadatas, ids = [], [], []
    for filename in sorted(os.listdir(dataset_folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(dataset_folder, filename), "r", encoding="utf-8") as f:
                content = f.read()
            chunks, current_chunk = [], ""
            for line in content.split("\n"):
                if len(current_chunk) + len(line) > 1000:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    current_chunk += "\n" + line
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk": j})
                ids.append(f"{filename}_chunk_{j}")

    embeddings = []
    for i in range(0, len(documents), 100):
        batch = documents[i:i + 100]
        for text in batch:
            embeddings.append(get_embedding(text))
        print(f"  Embedded {min(i + 100, len(documents))}/{len(documents)}")

    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    print(f"[Flat RAG] Saved {len(documents)} chunks\n")
    return collection


def flat_rag_query(collection, query: str) -> tuple:
    start = time.time()
    query_emb = get_embedding(query)
    results = collection.query(query_embeddings=[query_emb], n_results=5)
    context = "\n\n".join(results["documents"][0])
    response = mimo_client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {"role": "system", "content": "Answer based on context. If insufficient info, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0, max_tokens=500
    )
    elapsed = time.time() - start
    return response.choices[0].message.content, elapsed


def graph_rag_query(searcher: NodeSearch, query: str) -> tuple:
    start = time.time()
    try:
        result = searcher.answer(query)
    except Exception as e:
        result = f"Error: {e}"
    elapsed = time.time() - start
    return result, elapsed


BENCHMARK_QUESTIONS = [
    "What are the main electric vehicle manufacturers in the US market?",
    "How has Tesla's market share in the EV market changed from 2023 to 2024?",
    "What government policies support electric vehicle adoption in the United States?",
    "What are the environmental benefits of electric vehicles compared to conventional cars?",
    "How does charging infrastructure availability affect EV adoption rates?",
    "What is the average transaction price for a new electric vehicle in the US?",
    "Which states have the highest electric vehicle adoption rates?",
    "What role do zero-emission vehicle (ZEV) regulations play in EV market growth?",
    "How do federal tax credits impact electric vehicle sales?",
    "What are the main challenges facing electric vehicle battery technology?",
    "How does the life cycle emissions of EVs compare to conventional vehicles?",
    "What companies are investing the most in electric vehicle research and development?",
    "How has the EV charging network grown in the United States?",
    "What is the relationship between EV adoption and renewable energy sources?",
    "How do luxury EV brands compare to mass-market EV brands in sales?",
    "What impact does EV adoption have on the US petroleum consumption?",
    "How do state-level incentives affect electric vehicle purchases?",
    "What are the predictions for EV market share by 2030?",
    "How does the cost of owning an EV compare to a conventional vehicle over time?",
    "What role do automakers like Ford, GM, and Rivian play in the EV transition?",
]


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: Flat RAG vs GraphRAG - 20 Questions")
    print("=" * 80)

    dataset_folder = "C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY\\dataset"
    collection = build_flat_rag(dataset_folder)

    print("[GraphRAG] Loading NodeRAG...")
    config = NodeConfig.from_main_folder("C:\\Users\\admin\\Documents\\GitHub\\GRAPHRAG-with-TECHCOMPANY")
    searcher = NodeSearch(config)
    searcher.load_graph()
    searcher.load_hnsw()
    print("[GraphRAG] Ready!\n")

    results = []
    flat_total_time = 0
    graph_total_time = 0

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"[{i}/20] {question[:60]}...")
        flat_answer, flat_time = flat_rag_query(collection, question)
        graph_answer, graph_time = graph_rag_query(searcher, question)
        flat_total_time += flat_time
        graph_total_time += graph_time
        results.append({
            "id": i, "question": question,
            "flat_rag": flat_answer[:600], "flat_time": round(flat_time, 2),
            "graph_rag": graph_answer[:600] if isinstance(graph_answer, str) else str(graph_answer)[:600],
            "graph_time": round(graph_time, 2)
        })

    print("\nEvaluating with LLM...")
    eval_prompt = """Evaluate these 20 Flat RAG vs GraphRAG answers. For each, provide scores (1-5) and winner.

Return JSON array:
[{"id":1,"flat_score":X,"graph_score":X,"winner":"Flat/Graph/Tie","note":"brief reason"},...]

"""
    for r in results:
        eval_prompt += f"\nQ{r['id']}: {r['question']}\nFlat: {r['flat_rag'][:300]}\nGraph: {r['graph_rag'][:300]}\n"

    response = mimo_client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {"role": "system", "content": "RAG evaluator. Return only valid JSON array."},
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.0, max_tokens=4000
    )
    eval_text = response.choices[0].message.content

    try:
        if "```json" in eval_text:
            eval_text = eval_text.split("```json")[1].split("```")[0]
        elif "```" in eval_text:
            eval_text = eval_text.split("```")[1].split("```")[0]
        eval_results = json.loads(eval_text)
    except:
        eval_results = None

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS: Flat RAG vs GraphRAG")
    print("=" * 100)

    if eval_results:
        header = f"{'#':<3} {'Question':<50} {'Flat':<5} {'Graph':<5} {'Time(F)':<8} {'Time(G)':<8} {'Winner':<7} {'Note'}"
        print(header)
        print("-" * 120)

        flat_wins = graph_wins = ties = 0
        flat_total_score = graph_total_score = 0

        for ev in eval_results:
            r = next((x for x in results if x['id'] == ev['id']), None)
            if r:
                q = r['question'][:47] + "..." if len(r['question']) > 50 else r['question']
                w = ev.get('winner', '?')
                if 'Flat' in str(w): flat_wins += 1
                elif 'Graph' in str(w): graph_wins += 1
                else: ties += 1
                flat_total_score += ev.get('flat_score', 0)
                graph_total_score += ev.get('graph_score', 0)
                print(f"{ev['id']:<3} {q:<50} {ev.get('flat_score','?'):<5} {ev.get('graph_score','?'):<5} {r['flat_time']:<8} {r['graph_time']:<8} {w:<7} {ev.get('note','')[:30]}")

        print("-" * 120)
        n = len(eval_results)
        print(f"\nSUMMARY:")
        print(f"  Flat RAG wins:  {flat_wins}/20 ({flat_wins*100//20}%)")
        print(f"  GraphRAG wins:  {graph_wins}/20 ({graph_wins*100//20}%)")
        print(f"  Ties:           {ties}/20 ({ties*100//20}%)")
        print(f"  Avg Flat score: {flat_total_score/n:.2f}/5")
        print(f"  Avg Graph score:{graph_total_score/n:.2f}/5")
        print(f"  Flat RAG total time:  {flat_total_time:.1f}s (avg {flat_total_time/n:.2f}s/query)")
        print(f"  GraphRAG total time:  {graph_total_time:.1f}s (avg {graph_total_time/n:.2f}s/query)")

    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("BENCHMARK: Flat RAG vs GraphRAG - 20 Questions\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Q{r['id']}: {r['question']}\n")
            f.write(f"Flat RAG ({r['flat_time']}s): {r['flat_rag']}\n")
            f.write(f"GraphRAG ({r['graph_time']}s): {r['graph_rag']}\n")
            f.write("-" * 60 + "\n\n")
        f.write(f"\nEVALUATION:\n{eval_text}")

    print("\nSaved to: benchmark_results.txt")


if __name__ == "__main__":
    run_benchmark()
