import pandas as pd
from itertools import combinations

def compute_affinity_scores(votes: pd.DataFrame, season: int, smoothing: float = 1.0) -> pd.DataFrame:
    season_votes = votes[votes["season"] == season]
    pair_stats = {}

    for sog_id, grp in season_votes.groupby("sog_id"):
        votes_this_sog = grp.set_index("castaway_id")["vote_id"].to_dict()
        voters = list(votes_this_sog.keys())
        for a, b in combinations(sorted(voters), 2):
            stats = pair_stats.setdefault((a, b), [0, 0, 0])
            stats[0] += 1  # shared tribal councils
            if votes_this_sog[a] == votes_this_sog[b]:
                stats[1] += 1  # voted for the same person
            if votes_this_sog.get(a) == b or votes_this_sog.get(b) == a:
                stats[2] += 1  # voted against each other

    rows = []
    for (a, b), (shared, agree, against) in pair_stats.items():
        # Smoothing away from -1 and 1
        affinity = (agree - against) / (shared + smoothing)
        rows.append({
            "season": season, "castaway_a": a, "castaway_b": b,
            "shared_tribals": shared, "agree": agree, "against": against,
            "affinity": round(affinity, 3),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    votes = pd.read_csv("data/clean_votes.csv")
    scores = compute_affinity_scores(votes, season=40)
    print(scores.sort_values("affinity", ascending=False).head(8).to_string(index=False))