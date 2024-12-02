from flask import Flask, jsonify, request
from flask_cors import CORS
import heapq
import pandas as pd
import networkx as nx

app = Flask(__name__)
CORS(app)

# File paths for graph data
NODE_FILE = "data/citation_network_nodes2.csv"
EDGE_FILE = "data/citation_network_edges2.csv"


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
        # Calculate edge weight as 1 / (PR_source * PR_target)
        pr_source = G.nodes[source]['pagerank']
        pr_target = G.nodes[target]['pagerank']
        weight = 1 / (pr_source * pr_target)
        G.add_edge(source, target, weight=weight)
    
    return G

def filter_nodes_by_keyword(G, keyword):
    """
    Filter papers based on the keyword.
    """
    filtered_nodes = [
        node for node, data in G.nodes(data=True)
        if keyword.lower() in data['label'].lower()
    ]
    return filtered_nodes


def find_highest_pagerank_node(G, filtered_nodes):
    """
    Find the paper with the highest PageRank score in the filtered set.
    """
    pagerank_scores = nx.pagerank(G)
    highest_pagerank_node = max(
        filtered_nodes,
        key=lambda node: pagerank_scores.get(node, 0)
    )
    return highest_pagerank_node, pagerank_scores


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


@app.route('/api/filter', methods=['POST'])
def filter_by_keywords():
    data = request.get_json()
    keyword = data.get('keyword', '').strip()

    if not keyword:
        return jsonify({"success": False, "message": "No keyword provided."})

    try:
        # Load the graph
        graph = load_graph(NODE_FILE, EDGE_FILE)

        # Filter nodes by keyword
        filtered_nodes = filter_nodes_by_keyword(graph, keyword)

        if not filtered_nodes:
            return jsonify({"success": False, "message": "No papers found matching the keyword."})

        # Prepare node details
        nodes_details = [
            {
                "Id": node,
                "Label": graph.nodes[node]['label'],
                "Year": graph.nodes[node]['year'],
                "Authors": graph.nodes[node]['authors']
            }
            for node in filtered_nodes
        ]

        return jsonify({"success": True, "nodes": nodes_details})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/path', methods=['POST'])
def find_path():
    data = request.get_json()
    start_point = data.get('start_point')
    keyword = data.get('keyword', '').strip()

    if not start_point:
        return jsonify({"success": False, "message": "No start point provided."})

    if not keyword:
        return jsonify({"success": False, "message": "No keyword provided."})

    try:
        # Load the graph
        graph = load_graph(NODE_FILE, EDGE_FILE)

        # Filter nodes by keyword
        filtered_nodes = filter_nodes_by_keyword(graph, keyword)

        if not filtered_nodes:
            return jsonify({"success": False, "message": "No papers found matching the keyword."})

        # Find the paper with the highest PageRank score
        highest_pagerank_node, pagerank_scores = find_highest_pagerank_node(graph, filtered_nodes)

        # Generate the optimal learning path using Dijkstra's algorithm
        path, total_weight = dijkstra(graph, start_point, highest_pagerank_node)

        if path:
            # Prepare path details
            path_details = [
                {
                    "Id": node,
                    "Label": graph.nodes[node]['label'],
                    "Year": graph.nodes[node]['year'],
                    "Authors": graph.nodes[node]['authors']
                }
                for node in path
            ]

            return jsonify({
                "success": True,
                "path": path_details,
                "total_weight": total_weight,
                "highest_pagerank_node": {
                    "Id": highest_pagerank_node,
                    "Label": graph.nodes[highest_pagerank_node]['label'],
                    "Year": graph.nodes[highest_pagerank_node]['year'],
                    "Authors": graph.nodes[highest_pagerank_node]['authors']
                }
            })
        else:
            return jsonify({"success": False, "message": "No path exists between the start and end nodes."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
