import heapq
import pandas as pd
import networkx as nx

def load_graph(node_file, edge_file):
    """
    Load nodes and edges to create a directed graph.
    """
    G = nx.DiGraph()
    # Load node data  
    nodes_df = pd.read_csv(node_file)
    for _, row in nodes_df.iterrows():
        G.add_node(
            str(row['Id']),
            label=row['Label'],
            year=row['Year'],
            authors=row['Authors'],
            pagerank=row['PageRank']
        )
    
    # Load edge data
    edges_df = pd.read_csv(edge_file)
    for _, row in edges_df.iterrows():
        source, target = str(row['Source']), str(row['Target'])
        pr_source = G.nodes[source]['pagerank']
        pr_target = G.nodes[target]['pagerank']
        weight = 1 / (pr_source * pr_target)
        G.add_edge(source, target, weight=weight)
    return G

def filter_nodes_by_keyword(G, keyword):
    """
    Filter papers based on the keyword and display their titles and IDs.
    """
    filtered_nodes = [
        (node, data['label']) for node, data in G.nodes(data=True)
        if keyword.lower() in data['label'].lower()
    ]
    if filtered_nodes:
        print("\nPapers containing the keyword:")
        for node, label in filtered_nodes:
            print(f"  ID: {node}, Title: {label}")
    return [node for node, _ in filtered_nodes]

def bfs_reachable_nodes(G, start):
    """
    Find all reachable nodes from a starting node using BFS.
    """
    queue = [start]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            for neighbor in G.successors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
    print(visited)
    return visited

def find_highest_pagerank_in_reachable_nodes(G, reachable_nodes):
    """
    Find the paper with the highest PageRank score in the reachable nodes.
    """
    highest_pagerank_node = max(
        reachable_nodes,
        key=lambda node: G.nodes[node]['pagerank']
    )
    return highest_pagerank_node

def dijkstra(graph, start, end):
    """
    Implement Dijkstra's algorithm to find the shortest path between two nodes.
    """
    pq = [(0, start, [])]
    visited = set()
    
    while pq:
        current_weight, current_node, path = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        path = path + [current_node]
        
        if current_node == end:
            return path, current_weight
        
        for neighbor, edge_data in graph[current_node].items():
            if neighbor not in visited:
                weight = edge_data.get('weight', float('inf'))
                heapq.heappush(pq, (current_weight + weight, neighbor, path))
    
    return [], float('inf')

def display_path(G, path):
    """
    Display the recommended learning path with detailed information.
    """
    print("Recommended Learning Path:")
    for node in path:
        data = G.nodes[node]
        print(f"Paper {node}:\n"
              f"  Title: {data['label']}\n"
              f"  Authors: {data['authors']}\n"
              f"  Year: {data['year']}\n"
              f"  PageRank: {data['pagerank']}\n")

def main():
    # File paths
    node_file = "citation_network_nodes2.csv"
    edge_file = "citation_network_edges2.csv"
    
    # Load the graph
    G = load_graph(node_file, edge_file)
    
    # User input
    keyword = input("Enter a keyword: ").strip()
    
    # Filter papers by keyword
    filtered_nodes = filter_nodes_by_keyword(G, keyword)
    if not filtered_nodes:
        print("No papers found matching the keyword!")
        return
    
    # Let user choose the starting paper
    print("\nChoose a starting paper by entering its ID:")
    start_node = input("Enter the starting paper ID: ").strip()
    if start_node not in G.nodes:
        print(f"Paper ID {start_node} does not exist in the graph!")
        return
    
    # Find reachable nodes using BFS
    reachable_nodes = bfs_reachable_nodes(G, start_node)
    # Intersect with filtered nodes to narrow down search space
    
    if not reachable_nodes:
        print("No reachable papers match the keyword!")
        return
    
    # Find the paper with the highest PageRank score in the reachable nodes
    highest_pagerank_node = find_highest_pagerank_in_reachable_nodes(G, reachable_nodes)
    print(f"\nThe paper with the highest PageRank has ID: {highest_pagerank_node}")
    
    # Generate the optimal learning path using Dijkstra's algorithm
    path, total_weight = dijkstra(G, start_node, highest_pagerank_node)
    if path:
        display_path(G, path)
    else:
        print("\nNo path exists between the start and end nodes!")

if __name__ == "__main__":
    main()