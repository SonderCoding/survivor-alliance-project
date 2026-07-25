import pandas as pd

def load_clean_votes(path="data/raw/vote_history.csv") -> pd.DataFrame:
    vh = pd.read_csv(path)

    # Drop rows where no vote was cast - med-evac, quit, vote-steal etc
    vh = vh.dropna(subset=["castaway_id", "vote_id"])

    return vh

def load_outcomes(path="data/raw/castaways.csv") -> pd.DataFrame:
    cw = pd.read_csv(path)
    outcomes = cw[[
        "season", "castaway_id", "castaway", "result", "result_number",
        "jury", "finalist", "winner", "original_tribe"
    ]].drop_duplicates(subset=["season", "castaway_id"])
    return outcomes

if __name__ == "__main__":
    votes = load_clean_votes()
    outcomes = load_outcomes()
    votes.to_csv("data/clean_votes.csv", index=False)
    outcomes.to_csv("data/outcomes.csv", index=False)
    print(f"Clean votes: {votes.shape[0]} rows across {votes['season'].nunique()} seasons")