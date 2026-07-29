import streamlit as st
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict, Counter
import random
import numpy as np
from itertools import combinations

st.set_page_config(page_title="Survivor Alliance Network", layout="wide")

votes = pd.read_csv("data/raw/vote_history.csv").dropna(subset=["castaway_id", "vote_id"])
boot_mapping = pd.read_csv("data/raw/boot_mapping.csv")

version_options = sorted(votes["version"].unique())
version = st.selectbox("Show", version_options, index=version_options.index("US") if "US" in version_options else 0)

season_options = sorted(
    votes[votes.version == version]["season"].dropna().unique().astype(int), reverse=True
)
season = st.selectbox("Season", season_options)


def get_boot_order_lookup(bm, version, season):
    s = bm[(bm.version == version) & (bm.season == season)]
    return s[["sog_id", "order"]].drop_duplicates().set_index("sog_id")["order"].to_dict()


sog_to_order = get_boot_order_lookup(boot_mapping, version, season)
max_order = int(max(sog_to_order.values())) if sog_to_order else 1
up_to_order = st.slider("Up to elimination #", 1, max_order, 1)


def compute_affinity_scores(votes, version, season, sog_to_order, up_to_order=None, smoothing=1.0):
    sv = votes[(votes.version == version) & (votes.season == season)].copy()
    if up_to_order is not None:
        sv = sv[sv["sog_id"].map(sog_to_order) <= up_to_order]
    pair_stats = {}
    for sog_id, grp in sv.groupby("sog_id"):
        vt = grp.set_index("castaway_id")["vote_id"].to_dict()
        voters = list(vt.keys())
        for a, b in combinations(sorted(voters), 2):
            stats = pair_stats.setdefault((a, b), [0, 0, 0])
            stats[0] += 1
            if vt[a] == vt[b]:
                stats[1] += 1
            if vt.get(a) == b or vt.get(b) == a:
                stats[2] += 1
    rows = []
    for (a, b), (shared, agree, against) in pair_stats.items():
        affinity = (agree - against) / (shared + smoothing)
        rows.append({"castaway_a": a, "castaway_b": b, "affinity": round(affinity, 3)})
    return pd.DataFrame(rows)


def get_active_roster(bm, version, season, up_to_order):
    s = bm[(bm.version == version) & (bm.season == season)]
    in_game = s[s.game_status == "In the game"]
    last_order = in_game.groupby("castaway_id")["order"].max()
    active_ids = last_order[last_order >= up_to_order].index.tolist()
    roster = in_game[in_game.castaway_id.isin(active_ids) & (in_game.order <= up_to_order)]
    latest = roster.sort_values("order").groupby("castaway_id").tail(1)
    return latest[["castaway_id", "castaway", "tribe"]].reset_index(drop=True)


def resolve_overlaps(pos, min_dist=0.35, iterations=100):
    nodes = list(pos.keys())
    p = {n: list(v) for n, v in pos.items()}
    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                dx = p[b][0] - p[a][0]
                dy = p[b][1] - p[a][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_dist:
                    if dist < 1e-6:
                        dx, dy = random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
                        dist = 0.01
                    push = (min_dist - dist) / 2
                    ux, uy = dx / dist, dy / dist
                    p[a][0] -= ux * push; p[a][1] -= uy * push
                    p[b][0] += ux * push; p[b][1] += uy * push
                    moved = True
        if not moved:
            break
    return {n: tuple(v) for n, v in p.items()}


def get_stable_colors(members_by_cluster, version, season, threshold=0.4):
    """Matches each of today's clusters against previously-seen alliance
    identities (by member overlap) so the same real alliance keeps the
    same color across the slider, instead of getting a new random color
    every time Louvain happens to re-number its clusters."""
    key = f"{version}_{season}"
    if "cluster_identities" not in st.session_state:
        st.session_state.cluster_identities = {}
    if key not in st.session_state.cluster_identities:
        st.session_state.cluster_identities[key] = []
    registry = st.session_state.cluster_identities[key]

    color_assignment = {}
    used = set()
    next_idx = max([r["color_idx"] for r in registry], default=-1) + 1

    for cluster_id, members in members_by_cluster.items():
        mset = set(members)
        best, best_overlap = None, 0
        for r in registry:
            union = mset | r["members"]
            if not union:
                continue
            overlap = len(mset & r["members"]) / len(union)
            if overlap > best_overlap:
                best_overlap = overlap
                best = r
        if best and best_overlap >= threshold and best["color_idx"] not in used:
            color_assignment[cluster_id] = best["color_idx"]
            best["members"] = mset
            used.add(best["color_idx"])
        else:
            color_assignment[cluster_id] = next_idx
            registry.append({"members": mset, "color_idx": next_idx})
            used.add(next_idx)
            next_idx += 1

    return color_assignment


roster_df = get_active_roster(boot_mapping, version, season, up_to_order)
active_ids = set(roster_df["castaway_id"])
id_to_name = dict(zip(roster_df["castaway_id"], roster_df["castaway"]))

affinity_df = compute_affinity_scores(votes, version, season, sog_to_order, up_to_order)
affinity_df = affinity_df[
    affinity_df["castaway_a"].isin(active_ids) & affinity_df["castaway_b"].isin(active_ids)
]

# Players who haven't been to a single Tribal Council yet get pulled out of
# the physics simulation entirely, so they can't randomly land close to
# someone they have zero actual relationship data with
voted_ids = set(affinity_df["castaway_a"]) | set(affinity_df["castaway_b"]) if len(affinity_df) else set()
no_data_ids = active_ids - voted_ids
graphed_ids = active_ids - no_data_ids

G_full = nx.Graph()
G_full.add_nodes_from(graphed_ids)
for _, row in affinity_df.iterrows():
    G_full.add_edge(row["castaway_a"], row["castaway_b"], weight=row["affinity"])

G_positive = nx.Graph()
for _, row in affinity_df[affinity_df["affinity"] > 0].iterrows():
    G_positive.add_edge(row["castaway_a"], row["castaway_b"], weight=row["affinity"])

partition_raw = (
    community_louvain.best_partition(G_positive, weight="weight", random_state=42, resolution=1.2)
    if G_positive.number_of_edges() > 0
    else {}
)
cluster_sizes = Counter(partition_raw.values())
partition = {n: c for n, c in partition_raw.items() if cluster_sizes[c] >= 2}

G_layout = nx.Graph()
G_layout.add_nodes_from(graphed_ids)
for _, row in affinity_df.iterrows():
    a, b, w = row["castaway_a"], row["castaway_b"], row["affinity"]
    w = max(w, -0.3)
    if a in partition and b in partition:
        if partition[a] == partition[b] and w > 0:
            w = w * 1.8
        elif partition[a] != partition[b] and w > 0:
            w = w * 0.5
    G_layout.add_edge(a, b, weight=w)

pos = nx.spring_layout(G_layout, seed=42, weight="weight", k=0.9) if G_layout.number_of_nodes() > 0 else {}
pos = resolve_overlaps(pos, min_dist=0.35) if pos else {}

members_by_cluster = defaultdict(list)
for node, cluster_id in partition.items():
    members_by_cluster[cluster_id].append(node)

stable_colors = get_stable_colors(members_by_cluster, version, season)

fig, ax = plt.subplots(figsize=(11, 8))
cmap = plt.get_cmap("Set2")

SUPER_ALLIANCE_THRESHOLD = 0.65  # top ~10% tightness, based on real data across several seasons

for cluster_id, members in members_by_cluster.items():
    xs = [pos[m][0] for m in members]
    ys = [pos[m][1] for m in members]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    width = (max(xs) - min(xs)) + 0.4
    height = (max(ys) - min(ys)) + 0.4

    intra = affinity_df[
        affinity_df["castaway_a"].isin(members) & affinity_df["castaway_b"].isin(members)
    ]
    avg_affinity = intra["affinity"].mean() if len(intra) else 0.3
    alpha = min(0.15 + avg_affinity * 0.5, 0.55)
    is_super = avg_affinity >= SUPER_ALLIANCE_THRESHOLD
    color_idx = stable_colors[cluster_id]

    ellipse = Ellipse(
        (cx, cy), width, height, facecolor=cmap(color_idx % 8), alpha=alpha, zorder=0,
        edgecolor="#D4AF37" if is_super else "none", linewidth=3 if is_super else 0,
    )
    ax.add_patch(ellipse)


rivalry_edges = [(u, v) for u, v, d in G_full.edges(data=True) if d["weight"] < -0.5]
nx.draw_networkx_edges(G_full, pos, edgelist=rivalry_edges, edge_color="#B22222", style="dashed", width=1.5, ax=ax)

if graphed_ids:
    node_colors = [
        cmap(stable_colors[partition[n]] % 8) if n in partition else "#B0B0B0" for n in G_full.nodes
    ]
    nx.draw_networkx_nodes(G_full, pos, node_color=node_colors, node_size=650, edgecolors="white", linewidths=1.5, ax=ax)
    labels = {n: id_to_name.get(n, n) for n in G_full.nodes}
    nx.draw_networkx_labels(G_full, pos, labels=labels, font_size=9, ax=ax)

if pos:
    xs_all = [p[0] for p in pos.values()]
    ys_all = [p[1] for p in pos.values()]
    min_x, max_x = min(xs_all), max(xs_all)
    min_y, max_y = min(ys_all), max(ys_all)
else:
    min_x, max_x, min_y, max_y = -1, 1, -1, 1

if no_data_ids:
    sidebar_x = min_x - 1.3
    names_sorted = sorted([id_to_name.get(n, n) for n in no_data_ids])
    ax.text(sidebar_x, max_y + 0.3, "Haven't voted yet:", fontsize=10, fontweight="bold", va="center", ha="left")
    y_positions = np.linspace(max_y, min_y, len(names_sorted)) if len(names_sorted) > 1 else [(max_y + min_y) / 2]
    for y, nm in zip(y_positions, names_sorted):
        ax.text(sidebar_x, y, nm, fontsize=9, va="center", ha="left", color="#555555")
    ax.set_xlim(sidebar_x - 0.5, max_x + 0.6)

episode_lookup = boot_mapping[(boot_mapping.version == version) & (boot_mapping.season == season)]
current_ep_row = episode_lookup[episode_lookup["order"] == up_to_order]
episode_note = f" (Episode {int(current_ep_row['episode'].iloc[0])})" if len(current_ep_row) else ""

ax.set_title(f"{version} Season {season} — elimination #{up_to_order}{episode_note}")
ax.axis("off")

st.pyplot(fig)
st.caption(
    "Bubble color = inferred alliance (stable across the slider), gold border = a super tight alliance. "
    "Red dashed lines = strong rivalries. Names on the left haven't voted yet."
)