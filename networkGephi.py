import pandas as pd
import networkx as nx  
from collections import defaultdict
from pyvis.network import Network
import dataPreprocess as data

def create_citation_network(df):
    """
    Create an interactive citation network visualization
    """
    G = nx.DiGraph()
    
    paper_ids = set(str(pid) for pid in df['index'])
    print(f"Total unique papers: {len(paper_ids)}")

    edges_data = []

    # Add nodes and edges
    for _, paper in df.iterrows():
        paper_id = str(paper['index'])
        
        # Node information
        node_title = (f"Title: {paper['title']}\n"
                     f"Author: {paper['authors']}\n"
                     f"Year: {paper['year']}")

        # Add node
        G.add_node(paper_id, 
                  title=node_title,
                  label=f"Paper {paper_id}")
        

        try:
            if isinstance(paper['references'], str):
                try:
                    references = eval(paper['references'])
                except:
                    references = []
            elif isinstance(paper['references'], list):
                references = paper['references']
            else:
                references = []

            # add edges
            for ref in references:
                ref = str(ref).strip()  
                if ref and ref in paper_ids:  
                    G.add_edge(paper_id, ref)
                    edges_data.append({
                        'Source': paper_id,
                        'Target': ref,
                        'Type': 'Directed'
                    })
                    
        except Exception as e:
            print(f"Error processing references for paper {paper_id}: {e}")
            continue
    
    node_degrees = dict(G.degree())
    # nodes_to_remove = [node for node, degree in node_degrees.items() if degree <= 1]
    # G.remove_nodes_from(nodes_to_remove)

    # Remove isolated nodes
    initial_nodes = G.number_of_nodes()
    removed_count = remove_isolated_nodes(G)
    print(f"\nRemoved {removed_count} isolated nodes (nodes with no connections)")
    print(f"Nodes reduced from {initial_nodes} to {G.number_of_nodes()}")

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank = nx.pagerank(G)
    
    nodes_data = []
    for node in G.nodes():
        paper = df[df['index'] == int(node)].iloc[0]
        nodes_data.append({
            'Id': node,
            'Label': paper['title'] if pd.notnull(paper['title']) else f"Paper {node}",
            'Year': paper['year'] if pd.notnull(paper['year']) else '',
            'Authors': paper['authors'] if pd.notnull(paper['authors']) else '',
            'InDegree': in_degree.get(node, 0),
            'OutDegree': out_degree.get(node, 0),  
            'PageRank': pagerank.get(node, 0)
        })
    
    edges_data = [{'Source': e[0], 'Target': e[1], 'Type': 'Directed'} 
                  for e in G.edges()]
    
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)
    
    nodes_df.to_csv("citation_network_nodes1208_O.csv", index=False)
    edges_df.to_csv("citation_network_edges1208_O.csv", index=False)

    print("\nNetwork Statistics:")
    print(f"Number of nodes (papers): {G.number_of_nodes()}")
    print(f"Number of edges (citations): {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        print("\nTop cited papers (in-degree):")
        in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
        for node, degree in in_degrees:
            print(f"\nPaper {node} (cited {degree} times)")
            if 'title' in G.nodes[node]:
                print(f"Details: {G.nodes[node]['title']}")

        if len(in_degrees) > 0:
            max_citations = in_degrees[0][1]
            avg_citations = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
            print(f"\nCitation Metrics:")
            print(f"Maximum citations: {max_citations}")
            print(f"Average citations: {avg_citations:.2f}")

    return G


def remove_isolated_nodes(G):
    """
    Remove nodes with both in-degree and out-degree of 0
    Returns the number of nodes removed
    """
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    isolated_nodes = [node for node in G.nodes()
                     if in_degree[node] == 0 and out_degree[node] == 0]
    
    G.remove_nodes_from(isolated_nodes)
    return len(isolated_nodes)


def main():
    """
    Main function to read data and create network
    """
    try:
        file_path = "test.csv"
        print("Reading data from:", file_path)
        df = pd.read_csv(file_path)

        if df is not None:
            print(f"Successfully read {len(df)} papers")
            G = create_citation_network(df)
            return G
        else:
            print("Error: Could not read data from file")
            return None
            
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return None

if __name__ == "__main__":
    G = main()