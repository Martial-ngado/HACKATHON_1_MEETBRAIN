"""
Collaborative Filtering
Uses scipy.stats.pearsonr to find users with similar rating patterns,
then recommends movies they liked that the target user hasn't seen.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def build_rating_matrix(history_df: pd.DataFrame) -> pd.DataFrame:
    """Build user × movie rating matrix (NaN filled with 0)."""
    matrix = history_df.pivot_table(
        index="user_id", columns="title", values="user_rating"
    ).fillna(0)
    return matrix


def get_similar_users(target_id: str, rating_matrix: pd.DataFrame,
                      top_n: int = 8) -> list:
    """Return top-n users most correlated with target_id via Pearson r."""
    if target_id not in rating_matrix.index:
        return []
    target_vec = rating_matrix.loc[target_id].values
    correlations = {}

    for uid in rating_matrix.index:
        if uid == target_id:
            continue
        other_vec = rating_matrix.loc[uid].values
        mask = (target_vec != 0) & (other_vec != 0)
        if mask.sum() < 2:
            continue
        corr, p_val = pearsonr(target_vec[mask], other_vec[mask])
        if not np.isnan(corr):
            correlations[uid] = corr

    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]


def collaborative_recommend(user_id: str, rating_matrix: pd.DataFrame,
                             history_df: pd.DataFrame, users_df: pd.DataFrame,
                             top_n: int = 5):
    """Recommend movies via collaborative filtering for a user."""
    name = users_df[users_df["user_id"] == user_id]["name"].values
    name = name[0] if len(name) else user_id

    similar = get_similar_users(user_id, rating_matrix)
    if not similar:
        print(f"  Not enough data for collaborative filtering for {user_id}.")
        return

    seen = set(history_df[history_df["user_id"] == user_id]["title"])
    scores: dict = {}

    for sim_uid, corr_weight in similar:
        sim_ratings = history_df[history_df["user_id"] == sim_uid]
        for _, row in sim_ratings.iterrows():
            t = row["title"]
            if t not in seen:
                scores[t] = scores.get(t, 0) + corr_weight * row["user_rating"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n{'─'*55}")
    print(f"  🤝  Collaborative Recs for {name} ({user_id})")
    print(f"      Top similar users: {[u for u,_ in similar[:3]]}")
    print(f"{'─'*55}")
    if not ranked:
        print("  No new recommendations found.")
    else:
        for title, score in ranked:
            row = history_df[history_df["title"] == title].iloc[0]
            print(f"  ✦  {title:<35} [{row['genre']}]  score: {score:.2f}")
    print()
