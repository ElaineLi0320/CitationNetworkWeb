import heapq
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def analyze_paths(G, test_cases):
    """
    Perform path quality analysis for multiple test cases and visualize results.
    """
    results = []
    reachable_counts = []
    path_lengths = []
    avg_pageranks = []

    for start_node in test_cases:
        reachable_nodes = bfs_reachable_nodes(G, start_node)
        if not reachable_nodes:
            reachable_counts.append(0)
            path_lengths.append(0)
            avg_pageranks.append(0)
            results.append((start_node, 0, 0, 0))
            continue

        highest_pagerank_node = find_highest_pagerank_in_reachable_nodes(G, reachable_nodes)
        path, total_weight = dijkstra(G, start_node, highest_pagerank_node)
        avg_pagerank = sum(G.nodes[node]['pagerank'] for node in path) / len(path) if path else 0
        
        reachable_counts.append(len(reachable_nodes))
        path_lengths.append(len(path))
        avg_pageranks.append(avg_pagerank)
        results.append((start_node, len(reachable_nodes), len(path), avg_pagerank))

    # Split data into <5 and >=5 groups and plot histograms
    plot_clipped_histogram_split(
        reachable_counts, "Number of Reachable Papers", "Count", "Reachable Papers", clip_range=(0, 500), split_value=6
    )
    plot_clipped_histogram_split(
        path_lengths, "Path Length", "Count", "Path Lengths", clip_range=(0, 20), split_value=5
    )
    plot_clipped_histogram_split(
        avg_pageranks, "Average PageRank in Paths", "Count", "Average PageRank", clip_range=(0, 0.1)
    )

    return results

def plot_clipped_histogram_split(data, xlabel, ylabel, title, clip_range=None, split_value=5):
    """
    Plot two histograms for the given data: one for values below the split value and one for the rest.

    Args:
        data (list): The data to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the histogram.
        clip_range (tuple): Optional, specify (min, max) range to clip the histogram.
        split_value (float): The value to split the data into two groups.
    """
    # Clip the data if a clip range is provided
    if clip_range:
        data = [min(max(value, clip_range[0]), clip_range[1]) for value in data]
    
    # Split the data
    below_split = [value for value in data if value < split_value]
    above_split = [value for value in data if value >= split_value]

    # Plot data below the split value
    plt.figure(figsize=(16, 9))
    plt.hist(below_split, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (< {split_value})")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Plot data above or equal to the split value
    plt.figure(figsize=(16, 9))
    plt.hist(above_split, bins=10, color='orange', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (>= {split_value})")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

import heapq
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def analyze_paths(G, test_cases):
    """
    Perform path quality analysis for multiple test cases and visualize results.
    """
    results = []
    reachable_counts = []
    path_lengths = []
    avg_pageranks = []

    for start_node in test_cases:
        reachable_nodes = bfs_reachable_nodes(G, start_node)
        if not reachable_nodes:
            reachable_counts.append(0)
            path_lengths.append(0)
            avg_pageranks.append(0)
            results.append((start_node, 0, 0, 0))
            continue

        highest_pagerank_node = find_highest_pagerank_in_reachable_nodes(G, reachable_nodes)
        path, total_weight = dijkstra(G, start_node, highest_pagerank_node)
        avg_pagerank = sum(G.nodes[node]['pagerank'] for node in path) / len(path) if path else 0
        
        reachable_counts.append(len(reachable_nodes))
        path_lengths.append(len(path))
        avg_pageranks.append(avg_pagerank)
        results.append((start_node, len(reachable_nodes), len(path), avg_pagerank))

    # Split data into <5 and >=5 groups and plot histograms
    plot_clipped_histogram_split(
        reachable_counts, "Number of Reachable Papers", "Count", "Reachable Papers", clip_range=(0, 500), split_value=6
    )
    plot_clipped_histogram_split(
        path_lengths, "Path Length", "Count", "Path Lengths", clip_range=(0, 20), split_value=5
    )
    plot_clipped_histogram_split(
        avg_pageranks, "Average PageRank in Paths", "Count", "Average PageRank", clip_range=(0, 0.1)
    )

    return results

def plot_clipped_histogram_split(data, xlabel, ylabel, title, clip_range=None, split_value=5):
    """
    Plot two histograms for the given data: one for values below the split value and one for the rest.

    Args:
        data (list): The data to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the histogram.
        clip_range (tuple): Optional, specify (min, max) range to clip the histogram.
        split_value (float): The value to split the data into two groups.
    """
    # Clip the data if a clip range is provided
    if clip_range:
        data = [min(max(value, clip_range[0]), clip_range[1]) for value in data]
    
    # Split the data
    below_split = [value for value in data if value < split_value]
    above_split = [value for value in data if value >= split_value]

    # Plot data below the split value
    plt.figure(figsize=(16, 9))
    plt.hist(below_split, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (< {split_value})")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Plot data above or equal to the split value
    plt.figure(figsize=(16, 9))
    plt.hist(above_split, bins=10, color='orange', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (>= {split_value})")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_pagerank_comparison_split(G, results):
    """
    Plot two comparisons of PageRank distributions: one for values < 0.0005 and one for values >= 0.0005
    """
    all_pageranks = [G.nodes[node]['pagerank'] for node in G.nodes()]
    
    path_avg_pageranks = []
    for start_node, _, path_length, avg_pagerank in results:
        if path_length > 0: 
            path_avg_pageranks.append(avg_pagerank)

    def split_data(data, threshold=0.0005):
        below = [x for x in data if x < threshold]
        above = [x for x in data if x >= threshold]
        return below, above

    all_below, all_above = split_data(all_pageranks)
    path_below, path_above = split_data(path_avg_pageranks)

    plt.figure(figsize=(12, 8))
    x_ticks = [0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045, 0.00050]
    
    hist_all, _ = np.histogram(all_below, bins=x_ticks)
    hist_avg, _ = np.histogram(path_below, bins=x_ticks)
    
    x = np.arange(len(x_ticks)-1)
    width = 0.35
    
    plt.bar(x - width/2, hist_all, width, 
            label='PageRank Distribution (< 0.0005)', 
            color='lightblue', 
            edgecolor='black')
    plt.bar(x + width/2, hist_avg, width, 
            label='Average PageRank (< 0.0005)', 
            color='orange', 
            edgecolor='black')
    
    plt.xticks(x, [f'{val:.5f}' for val in x_ticks[:-1]])
    plt.xlabel('PageRank Score')
    plt.ylabel('Count')
    plt.ylim(0, 3500)
    plt.grid(axis='y', alpha=0.3, linestyle='-')
    plt.title('Comparison of PageRank Distribution and Average PageRank (< 0.0005)', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    max_pagerank = max(max(all_pageranks), max(path_avg_pageranks))
    x_ticks_above = np.linspace(0.0005, max_pagerank, 8)
    
    hist_all, _ = np.histogram(all_above, bins=x_ticks_above)
    hist_avg, _ = np.histogram(path_above, bins=x_ticks_above)
    
    x = np.arange(len(x_ticks_above)-1)
    
    plt.bar(x - width/2, hist_all, width, 
            label='PageRank Distribution (>= 0.0005)', 
            color='lightblue', 
            edgecolor='black')
    plt.bar(x + width/2, hist_avg, width, 
            label='Average PageRank (>= 0.0005)', 
            color='orange', 
            edgecolor='black')
    
    plt.xticks(x, [f'{val:.5f}' for val in x_ticks_above[:-1]])
    plt.xlabel('PageRank Score')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3, linestyle='-')
    plt.title('Comparison of PageRank Distribution and Average PageRank (>= 0.0005)', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # File paths
    node_file = "/Users/hanwenwen/Downloads/CitationNetworkWeb-main/backend/citation_network_nodes1208_O.csv"
    edge_file = "/Users/hanwenwen/Downloads/CitationNetworkWeb-main/backend/citation_network_edges1208_O.csv"
    
    # Load the graph
    G = load_graph(node_file, edge_file)

    
    # Extract all paper IDs from the graph
    test_cases = list(G.nodes)  # Dynamically extract all node IDs
    print(f"Total Papers (Nodes): {len(test_cases)}")
    
    # Perform path quality analysis and display histograms
    results = analyze_paths(G, test_cases)

    plot_pagerank_comparison_split(G, results)

    
    # Example case study for a single paper
    if test_cases:
        start_node = test_cases[0]
        print(f"\nCase Study for Paper ID: {start_node}")
        reachable_nodes = bfs_reachable_nodes(G, start_node)
        if reachable_nodes:
            highest_pagerank_node = find_highest_pagerank_in_reachable_nodes(G, reachable_nodes)
            path, total_weight = dijkstra(G, start_node, highest_pagerank_node)
            print(f"Path to highest PageRank paper (ID {highest_pagerank_node}):")
            print(path)

if __name__ == "__main__":
    main()
