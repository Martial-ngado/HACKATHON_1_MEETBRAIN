"""
Content-Based Filtering
Recommends movies similar to ones the user already liked,
using TF-IDF on genre+director and cosine similarity.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(movies_df: pd.DataFrame):
    """Build a cosine-similarity matrix from movie genre + director features."""
    movies_df = movies_df.copy()
    movies_df["features"] = movies_df["genre"].str.lower() + " " + movies_df["director"].str.lower()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_df["features"])
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim


def recommend_similar_movies(title: str, movies_df: pd.DataFrame,
                              cosine_sim, n: int = 5) -> pd.DataFrame:
    """Return top-n movies most similar to `title`."""
    matches = movies_df[movies_df["title"].str.lower() == title.lower()]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]
    return movies_df.iloc[[s[0] for s in scores]][["title", "genre", "director", "avg_rating"]]


def recommend_for_user(user_id: str, users_df: pd.DataFrame,
                       movies_df: pd.DataFrame, history_df: pd.DataFrame,
                       cosine_sim, top_n: int = 5):
    """Generate content-based recommendations for a user."""
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        print(f"User {user_id} not found.")
        return

    user = row.iloc[0]
    name = user["name"]
    prefs = [g.strip().lower() for g in user["preferred_genres"].split("|")]

    # Movies this user rated highly
    user_hist = history_df[history_df["user_id"] == user_id]
    liked = user_hist[user_hist["user_rating"] >= 4]["title"].tolist()
    seen  = set(user_hist["title"].tolist())

    candidates = {}
    for movie_title in liked:
        recs = recommend_similar_movies(movie_title, movies_df, cosine_sim, n=10)
        for _, r in recs.iterrows():
            t = r["title"]
            if t not in seen:
                candidates[t] = max(candidates.get(t, 0), r["avg_rating"])

    # Filter to preferred genres
    final = movies_df[movies_df["title"].isin(candidates.keys())]
    final = final[final["genre"].str.lower().isin(prefs)]
    final = final.sort_values("avg_rating", ascending=False).head(top_n)

    print(f"\n{'─'*55}")
    print(f"  🎬  Content-Based Recs for {name} ({user_id})")
    print(f"{'─'*55}")
    if final.empty:
        print("  No new recommendations found (try broader preferences).")
    else:
        for _, r in final.iterrows():
            print(f"  ✦  {r['title']:<35} [{r['genre']}]  ★ {r['avg_rating']}")
    print()
