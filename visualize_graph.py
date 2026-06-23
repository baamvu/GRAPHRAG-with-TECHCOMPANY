import sys
import io
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def visualize_graph():
    print("Loading graph...")
    with open("C:/Users/admin/Documents/GitHub/GRAPHRAG-with-TECHCOMPANY/cache/graph.pkl", "rb") as f:
        G = pickle.load(f)

    print(f"Graph stats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
    print(f"  Node types: {node_types}")

    max_nodes = 150
    if G.number_of_nodes() > max_nodes:
        nodes_by_weight = sorted(G.nodes(data=True), key=lambda x: x[1].get('weight', 0), reverse=True)
        top_nodes = [n[0] for n in nodes_by_weight[:max_nodes]]
        G_sub = G.subgraph(top_nodes).copy()
        print(f"\nVisualizing top {max_nodes} nodes by weight")
    else:
        G_sub = G

    color_map = {
        'entity': '#4ECDC4',
        'semantic_unit': '#FF6B6B',
        'relationship': '#45B7D1',
        'high_level_element': '#96CEB4',
        'high_level_element_title': '#FFEAA7'
    }

    node_colors = []
    node_sizes = []
    for node in G_sub.nodes(data=True):
        ntype = node[1].get('type', 'unknown')
        node_colors.append(color_map.get(ntype, '#888888'))
        weight = node[1].get('weight', 1)
        node_sizes.append(min(50 + weight * 10, 300))

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)

    nx.draw_networkx_edges(G_sub, pos, alpha=0.15, edge_color='#888888', width=0.5)
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    labels = {}
    for node in G_sub.nodes(data=True):
        context = node[1].get('context', '')
        if context and len(context) > 3:
            labels[node[0]] = context[:25] + "..." if len(context) > 25 else context

    nx.draw_networkx_labels(G_sub, pos, labels, font_size=6, font_weight='bold')

    legend_elements = [plt.scatter([], [], c=color, s=100, label=ntype) 
                       for ntype, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.title(f"Tech Company Knowledge Graph\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    output_path = "C:/Users/admin/Documents/GitHub/GRAPHRAG-with-TECHCOMPANY/knowledge_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to: {output_path}")

    plt.figure(figsize=(12, 8))
    type_counts = list(node_types.items())
    types = [t[0] for t in type_counts]
    counts = [t[1] for t in type_counts]
    colors = [color_map.get(t, '#888888') for t in types]
    
    plt.bar(types, counts, color=colors)
    plt.title("Node Type Distribution", fontsize=14)
    plt.xlabel("Node Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    stats_path = "C:/Users/admin/Documents/GitHub/GRAPHRAG-with-TECHCOMPANY/graph_stats.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"Stats saved to: {stats_path}")

if __name__ == "__main__":
    visualize_graph()
