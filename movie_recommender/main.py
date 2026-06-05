"""
main.py  —  AI-Powered Movie Recommendation System
============================================================
Topic 3 Hackathon Project
Combines:
  1. Content-Based Filtering (TF-IDF + cosine similarity)
  2. Collaborative Filtering (SciPy Pearson correlation)
  3. K-Means Clustering (SciPy cluster.vq)
  4. Visualizations (Matplotlib / Seaborn)
  5. Optional SciPy stats (chi-square, Spearman, t-test)
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

# ── local modules ─────────────────────────────────────────
from recommender.content_based  import build_similarity_matrix, recommend_for_user
from recommender.collaborative  import build_rating_matrix, collaborative_recommend
from recommender.clustering     import cluster_users, cluster_recommendations
from recommender.scipy_stats    import test_genre_bias, spearman_similarity, compare_cluster_ratings
from visualizations.plots       import (plot_genre_pie, plot_ratings_over_time,
                                        plot_platform_genres, plot_cluster_heatmap)

# ── paths ─────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────
print("\n⏳  Loading datasets...")
movies_df  = pd.read_csv(os.path.join(DATA, "movies.csv"))
history_df = pd.read_csv(os.path.join(DATA, "viewing_history.csv"))
users_df   = pd.read_csv(os.path.join(DATA, "users.csv"))
print(f"  ✅  {len(movies_df)} movies | {len(users_df)} users | {len(history_df)} history records")

# ── 2. Build models ───────────────────────────────────────
print("\n⏳  Building recommendation models...")
cosine_sim    = build_similarity_matrix(movies_df)
rating_matrix = build_rating_matrix(history_df)
users_df      = cluster_users(users_df, history_df, k=4)
print("  ✅  Models ready.")

# ── 3. Demo: run for 3 sample users ───────────────────────
SAMPLE_USERS = ["U001", "U010", "U025"]

print("\n" + "═"*55)
print("   🎬  PERSONALIZED MOVIE RECOMMENDATIONS")
print("═"*55)

for uid in SAMPLE_USERS:
    row  = users_df[users_df["user_id"] == uid]
    name = row["name"].values[0] if not row.empty else uid

    # ── Content-based ────────────────────────────────────
    recommend_for_user(uid, users_df, movies_df, history_df, cosine_sim)

    # ── Collaborative filtering ──────────────────────────
    collaborative_recommend(uid, rating_matrix, history_df, users_df)

    # ── Cluster-based ────────────────────────────────────
    cluster_recommendations(uid, users_df, history_df)

# ── 4. Visualizations ────────────────────────────────────
print("\n⏳  Generating visualizations...")
TARGET = "U001"
trow   = users_df[users_df["user_id"] == TARGET].iloc[0]
tname  = trow["name"]

plot_genre_pie(TARGET, history_df, name=tname, output_dir=OUT_DIR)
plot_ratings_over_time(TARGET, history_df, name=tname, output_dir=OUT_DIR)
plot_platform_genres(history_df, output_dir=OUT_DIR)
plot_cluster_heatmap(users_df, history_df, output_dir=OUT_DIR)

# ── 5. Optional SciPy stats ───────────────────────────────
print("\n" + "═"*55)
print("   📐  OPTIONAL SCIPY STATISTICAL ANALYSIS")
print("═"*55)

test_genre_bias("U001", history_df)
spearman_similarity("U001", "U010", rating_matrix)
compare_cluster_ratings(0, 1, users_df, history_df)

print("\n" + "═"*55)
print("   ✅  All done! Visualizations saved to /outputs/")
print("═"*55 + "\n")
