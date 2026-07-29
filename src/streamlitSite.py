# app.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict
from affinity import compute_affinity_scores
from graphSetup import build_affinity_graph, detect_communities

st.set_page_config(page_title="Survivor Alliance Network", layout="wide")

votes = pd.read_csv("data/clean_votes.csv")
boot_mapping = pd.read_csv("data/raw/boot_mapping.csv")

# Season list
season_options = sorted(votes["season"].dropna().unique().astype(int), reverse=True)
season = st.selectbox("Season", season_options)

max_episode = int(votes[votes.season == season]["episode"].max())
episode = st.slider("Up to episode", 1, max_episode, 1)

def get_active_roster(bm, season, up_to_episode):
    s = bm[bm["season"] == season]
    in_game = s[s["game_status"] == "In the game"]
    last_episode = in_game.groupby("castaway_id")["episode"].max()
    active_ids = last_episode[last_episode >= up_to_episode].index.tolist()
    roster = in_game[
        in_game["castaway_id"].isin(active_ids) & (in_game["episode"] <= up_to_episode)
    ]
    latest = roster.sort_values("episode").groupby("castaway_id").tail(1)
    return latest[["castaway_id", "castaway", "tribe"]].reset_index(drop=True)

# Only players still actually in the game appear at all
roster_df = get_active_roster(boot_mapping, season, episode)
active_ids = set(roster_df["castaway_id"])
id_to_name = dict(zip(roster_df["castaway_id"], roster_df["castaway"]))

filtered_votes = votes[(votes.season == season) & (votes.episode <= episode)]
affinity_df = compute_affinity_scores(filtered_votes, season)
affinity_df = affinity_df[
    affinity_df["castaway_a"].isin(active_ids) & affinity_df["castaway_b"].isin(active_ids)
]

# Every active player becomes a node first, even with zero edges
# (this is what keeps pre-merge, not-yet-at-Tribal players visible)
G_full = nx.Graph()
G_full.add_nodes_from(active_ids)
for _, row in affinity_df.iterrows():
    G_full.add_edge(row["castaway_a"], row["castaway_b"], weight=row["affinity"])

pos = nx.spring_layout(G_full, seed=42, weight="weight", k=0.8)

G_positive = build_affinity_graph(affinity_df)
partition = detect_communities(G_positive)

fig, ax = plt.subplots(figsize=(10, 8))

# Draws a translucent bubble behind each inferred alliance instead of
# drawing every agreement line, this is what makes grouping readable
members_by_cluster = defaultdict(list)
for node, cluster_id in partition.items():
    members_by_cluster[cluster_id].append(node)

cmap = plt.get_cmap("Set2")
for cluster_id, members in members_by_cluster.items():
    if len(members) < 2:
        continue
    xs = [pos[m][0] for m in members]
    ys = [pos[m][1] for m in members]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    width = (max(xs) - min(xs)) + 0.35
    height = (max(ys) - min(ys)) + 0.35
    ellipse = Ellipse((cx, cy), width, height, facecolor=cmap(cluster_id % 8), alpha=0.25, zorder=0)
    ax.add_patch(ellipse)

# Only strong rivalries get drawn as explicit lines, alliances are
# already shown by the bubbles
rivalry_edges = [(u, v) for u, v, d in G_full.edges(data=True) if d["weight"] < -0.5]
nx.draw_networkx_edges(G_full, pos, edgelist=rivalry_edges, edge_color="#B22222", style="dashed", width=1.5, ax=ax)

node_colors = [cmap(partition[n] % 8) if n in partition else "#B0B0B0" for n in G_full.nodes]
nx.draw_networkx_nodes(G_full, pos, node_color=node_colors, node_size=600, edgecolors="white", linewidths=1, ax=ax)

# Real names, using the boot_mapping lookup so even players with no
# votes cast yet still get a label
labels = {n: id_to_name.get(n, n) for n in G_full.nodes}
nx.draw_networkx_labels(G_full, pos, labels=labels, font_size=9, ax=ax)

ax.set_title(f"Season {season} — through episode {episode}")
ax.axis("off")

st.pyplot(fig)
st.caption(
    "Colored bubbles = inferred alliances. Red dashed lines = strong rivalries "
    "(affinity below -0.5). Gray nodes have no strong alliance yet."
)