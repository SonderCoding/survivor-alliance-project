import networkx as nx
import community as community_louvain

def build_affinity_graph(affinity_df) -> nx.Graph:
    G = nx.Graph()
    # Only positive go into the clustering graph
    for _, row in affinity_df[affinity_df["affinity"] > 0].iterrows():
        G.add_edge(row["castaway_a"], row["castaway_b"], weight=row["affinity"])
    return G

def detect_communities(G) -> dict:
    return community_louvain.best_partition(G, weight="weight", random_state=42)

def add_centrality(G: nx.Graph) -> dict:
    return {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight", max_iter=1000),
    }

if __name__ == "__main__":
    import pandas as pd
    from src.affinity import compute_affinity_scores

    votes = pd.read_csv("data/clean_votes.csv")
    affinity_df = compute_affinity_scores(votes, season=40)
    G = build_affinity_graph(affinity_df)
    partition = detect_communities(G)

    names = votes[votes["season"] == 40][["castaway_id", "castaway"]] \
        .drop_duplicates().set_index("castaway_id")["castaway"].to_dict()

    for cluster_id in sorted(set(partition.values())):
        members = [names.get(cid, cid) for cid, c in partition.items() if c == cluster_id]
        print(cluster_id, members)