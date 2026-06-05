"""
Cluster-Based Recommendations
Uses scipy.cluster.vq.kmeans to group users by genre rating patterns,
then recommends top-rated movies from the same cluster.
"""
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

GENRES = ["Action", "Drama", "Comedy", "Sci-Fi", "Horror",
          "Romance", "Thriller", "Animation", "Documentary", "Fantasy"]


def _user_genre_vector(user_id: str, history_df: pd.DataFrame) -> list:
    user_hist = history_df[history_df["user_id"] == user_id]
    return [
        user_hist[user_hist["genre"] == g]["user_rating"].mean()
        if len(user_hist[user_hist["genre"] == g]) > 0 else 0.0
        for g in GENRES
    ]


def cluster_users(users_df: pd.DataFrame, history_df: pd.DataFrame,
                  k: int = 4) -> pd.DataFrame:
    """Assign each user to one of k clusters based on genre rating profile."""
    users_df = users_df.copy()
    matrix = np.array([_user_genre_vector(uid, history_df)
                       for uid in users_df["user_id"]])

    # Replace NaN with 0
    matrix = np.nan_to_num(matrix)

    # Whiten (normalise) before kmeans
    whitened = whiten(matrix + 1e-9)   # add tiny value to avoid zero std

    centroids, _ = kmeans(whitened, k, seed=42)
    labels, _    = vq(whitened, centroids)

    users_df["cluster"] = labels
    return users_df


def cluster_recommendations(user_id: str, users_df: pd.DataFrame,
                             history_df: pd.DataFrame, top_n: int = 5):
    """Recommend top movies from the user's cluster."""
    row = users_df[users_df["user_id"] == user_id]
    if row.empty or "cluster" not in row.columns:
        print(f"  Cluster not assigned for {user_id}.")
        return

    user     = row.iloc[0]
    name     = user["name"]
    cluster  = user["cluster"]

    members  = users_df[users_df["cluster"] == cluster]["user_id"]
    cluster_hist = history_df[history_df["user_id"].isin(members)]

    avg_ratings = (cluster_hist.groupby("title")["user_rating"]
                               .mean().sort_values(ascending=False))

    seen = set(history_df[history_df["user_id"] == user_id]["title"])
    recs = avg_ratings[~avg_ratings.index.isin(seen)].head(top_n)

    print(f"\n{'─'*55}")
    print(f"  📊  Cluster-Based Recs for {name} ({user_id})")
    print(f"      Cluster #{cluster}  |  {len(members)} similar users")
    print(f"{'─'*55}")
    for title, avg in recs.items():
        genre = history_df[history_df["title"] == title]["genre"].values
        g = genre[0] if len(genre) else "?"
        print(f"  ✦  {title:<35} [{g}]  cluster avg ★ {avg:.2f}")
    print()
