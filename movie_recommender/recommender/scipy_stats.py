"""
Optional SciPy Statistical Analysis
- Chi-square test: genre preference bias
- Spearman correlation: rank-based user similarity
- T-test: compare rating distributions between clusters
"""
import numpy as np
import pandas as pd
from scipy.stats import chisquare, spearmanr, ttest_ind


def test_genre_bias(user_id: str, history_df: pd.DataFrame):
    """Chi-square test: does the user prefer certain genres significantly?"""
    user_hist = history_df[history_df["user_id"] == user_id]
    observed  = user_hist["genre"].value_counts().values

    if len(observed) < 2:
        print("  Not enough genre variety for chi-square test.")
        return

    expected = [len(user_hist) / len(observed)] * len(observed)
    stat, p  = chisquare(f_obs=observed, f_exp=expected)

    print(f"\n  📐 Chi-Square Test — {user_id}")
    print(f"     stat={stat:.3f}  p={p:.4f}")
    if p < 0.05:
        top_genre = user_hist["genre"].value_counts().index[0]
        print(f"     → Significant preference bias detected! Favourite: {top_genre}")
    else:
        print(f"     → No significant genre preference bias.")


def spearman_similarity(user_a: str, user_b: str, rating_matrix: pd.DataFrame):
    """Spearman rank correlation between two users."""
    if user_a not in rating_matrix.index or user_b not in rating_matrix.index:
        return None, None
    va   = rating_matrix.loc[user_a].values
    vb   = rating_matrix.loc[user_b].values
    mask = (va != 0) & (vb != 0)
    if mask.sum() < 3:
        return None, None
    corr, pval = spearmanr(va[mask], vb[mask])
    print(f"\n  📐 Spearman Correlation: {user_a} vs {user_b}")
    print(f"     r={corr:.3f}  p={pval:.4f}")
    return corr, pval


def compare_cluster_ratings(cluster_a: int, cluster_b: int,
                              users_df: pd.DataFrame, history_df: pd.DataFrame):
    """T-test: are average ratings different between two clusters?"""
    members_a = users_df[users_df["cluster"] == cluster_a]["user_id"]
    members_b = users_df[users_df["cluster"] == cluster_b]["user_id"]

    ratings_a = history_df[history_df["user_id"].isin(members_a)]["user_rating"]
    ratings_b = history_df[history_df["user_id"].isin(members_b)]["user_rating"]

    stat, p = ttest_ind(ratings_a, ratings_b)

    print(f"\n  📐 T-Test: Cluster {cluster_a} vs Cluster {cluster_b}")
    print(f"     Cluster {cluster_a} mean={ratings_a.mean():.2f}  "
          f"| Cluster {cluster_b} mean={ratings_b.mean():.2f}")
    print(f"     stat={stat:.3f}  p={p:.4f}")
    if p < 0.05:
        print(f"     → Significant rating difference between clusters!")
    else:
        print(f"     → No significant difference.")
