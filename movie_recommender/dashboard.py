"""
dashboard.py — AI Movie Recommendation Dashboard
Run with:  streamlit run dashboard.py
"""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark cinema theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: #0a0b0f; color: #e8e6e0; }
section[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #e8e6e0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #13151f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #7a7a8a !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e50914 !important; font-family: 'Bebas Neue', sans-serif; font-size: 2rem !important; }

/* Headers */
h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 3rem !important; color: #e50914 !important; letter-spacing: 2px; }
h2 { font-family: 'Bebas Neue', sans-serif !important; color: #f5c518 !important; letter-spacing: 1px; }
h3 { color: #e8e6e0 !important; font-weight: 500 !important; }

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #13151f !important;
    border-color: #1e2130 !important;
    color: #e8e6e0 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #7a7a8a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e50914 !important;
    border-bottom: 2px solid #e50914 !important;
}

/* Dataframe */
.stDataFrame { background: #13151f !important; }

/* Divider */
hr { border-color: #1e2130 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0b0f; }
::-webkit-scrollbar-thumb { background: #e50914; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Dark matplotlib style ─────────────────────────────────────────────────────
DARK   = "#0a0b0f"
CARD   = "#13151f"
RED    = "#e50914"
GOLD   = "#f5c518"
TEXT   = "#e8e6e0"
MUTED  = "#7a7a8a"
COLORS = [RED, GOLD, "#3b82f6", "#22c55e", "#a855f7",
          "#f97316", "#06b6d4", "#ec4899", "#84cc16", "#eab308"]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": "#1e2130", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#1e2130",
    "grid.alpha": 0.4, "font.family": "sans-serif",
})

# ── Data paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies  = pd.read_csv(os.path.join(DATA, "movies.csv"))
    history = pd.read_csv(os.path.join(DATA, "viewing_history.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    history["watch_date"] = pd.to_datetime(history["watch_date"])
    return users, movies, history

@st.cache_data
def build_models(movies_key, history_key):
    users_df, movies_df, history_df = load_data()

    # TF-IDF similarity
    movies_df["features"] = movies_df["genre"].str.lower() + " " + movies_df["director"].str.lower()
    tfidf  = TfidfVectorizer()
    matrix = tfidf.fit_transform(movies_df["features"])
    sim    = cosine_similarity(matrix, matrix)

    # Rating matrix
    rating_matrix = history_df.pivot_table(
        index="user_id", columns="title", values="user_rating"
    ).fillna(0)

    # K-Means clustering
    GENRES = ["Action","Drama","Comedy","Sci-Fi","Horror",
              "Romance","Thriller","Animation","Documentary","Fantasy"]
    def genre_vec(uid):
        uh = history_df[history_df["user_id"] == uid]
        return [uh[uh["genre"]==g]["user_rating"].mean() if len(uh[uh["genre"]==g])>0 else 0.0 for g in GENRES]

    feat = np.nan_to_num(np.array([genre_vec(uid) for uid in users_df["user_id"]]))
    wh   = whiten(feat + 1e-9)
    centroids, _ = kmeans(wh, 4, seed=42)
    labels, _    = vq(wh, centroids)
    users_df = users_df.copy()
    users_df["cluster"] = labels

    return sim, rating_matrix, users_df

users_df, movies_df, history_df = load_data()
cosine_sim, rating_matrix, users_df = build_models(len(movies_df), len(history_df))

# ── Recommendation functions ──────────────────────────────────────────────────
def is_new_user(user_id):
    """True if user has no watch history."""
    return len(history_df[history_df["user_id"] == user_id]) == 0

def content_recs(user_id, n=6):
    row   = users_df[users_df["user_id"] == user_id].iloc[0]
    prefs = [g.strip().lower() for g in row["preferred_genres"].split("|")]

    # NEW USER — no watch history: return top-rated movies in preferred genres
    if is_new_user(user_id):
        result = movies_df[movies_df["genre"].str.lower().isin(prefs)]
        result = result.sort_values("avg_rating", ascending=False).head(n).copy()
        result["rec_reason"] = "Top rated in your preferred genres"
        return result

    # EXISTING USER — use similarity from liked movies
    liked = history_df[(history_df["user_id"]==user_id) & (history_df["user_rating"]>=4)]["title"].tolist()
    seen  = set(history_df[history_df["user_id"]==user_id]["title"])
    candidates = {}
    for title in liked:
        m = movies_df[movies_df["title"].str.lower()==title.lower()]
        if m.empty: continue
        idx    = m.index[0]
        scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
        for i, sc in scores[1:11]:
            t = movies_df.iloc[i]["title"]
            if t not in seen:
                candidates[t] = max(candidates.get(t,0), sc)

    final = movies_df[movies_df["title"].isin(candidates)]
    final = final[final["genre"].str.lower().isin(prefs)]
    if final.empty:
        # Fallback: top rated in preferred genres not yet seen
        final = movies_df[movies_df["genre"].str.lower().isin(prefs)]
        final = final[~final["title"].isin(seen)]
    return final.sort_values("avg_rating", ascending=False).head(n)

def collab_recs(user_id, n=6):
    row   = users_df[users_df["user_id"] == user_id].iloc[0]
    prefs = [g.strip().lower() for g in row["preferred_genres"].split("|")]

    # NEW USER — no ratings yet: find users with same preferred genres, return their top picks
    if is_new_user(user_id):
        similar_users = []
        for _, u in users_df[users_df["user_id"] != user_id].iterrows():
            u_prefs = set(g.strip().lower() for g in u["preferred_genres"].split("|"))
            overlap = len(u_prefs & set(prefs))
            if overlap > 0:
                similar_users.append((u["user_id"], overlap))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[:10]
        scores = {}
        for sim_uid, weight in similar_users:
            for _, r in history_df[history_df["user_id"]==sim_uid].iterrows():
                if r["user_rating"] >= 4:
                    scores[r["title"]] = scores.get(r["title"], 0) + weight * r["user_rating"]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        if not ranked:
            return pd.DataFrame()
        result = movies_df[movies_df["title"].isin([t for t,_ in ranked])].copy()
        result["score"] = result["title"].map(dict(ranked))
        result["rec_reason"] = "Loved by users with similar genre taste"
        return result.sort_values("score", ascending=False)

    # EXISTING USER — Pearson correlation on rating vectors
    if user_id not in rating_matrix.index:
        return pd.DataFrame()
    tv  = rating_matrix.loc[user_id].values
    cors = {}
    for uid in rating_matrix.index:
        if uid == user_id: continue
        ov   = rating_matrix.loc[uid].values
        mask = (tv!=0) & (ov!=0)
        if mask.sum() < 2: continue
        c, _ = pearsonr(tv[mask], ov[mask])
        if not np.isnan(c): cors[uid] = c

    top_users = sorted(cors.items(), key=lambda x: x[1], reverse=True)[:8]
    seen      = set(history_df[history_df["user_id"]==user_id]["title"])
    scores    = {}
    for sim_uid, w in top_users:
        for _, r in history_df[history_df["user_id"]==sim_uid].iterrows():
            if r["title"] not in seen:
                scores[r["title"]] = scores.get(r["title"],0) + w * r["user_rating"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    if not ranked: return pd.DataFrame()
    result = movies_df[movies_df["title"].isin([t for t,_ in ranked])].copy()
    result["score"] = result["title"].map(dict(ranked))
    return result.sort_values("score", ascending=False)

def cluster_recs(user_id, n=6):
    row     = users_df[users_df["user_id"]==user_id].iloc[0]
    cluster = row["cluster"]
    members = users_df[users_df["cluster"]==cluster]["user_id"]
    ch      = history_df[history_df["user_id"].isin(members)]
    prefs   = [g.strip().lower() for g in row["preferred_genres"].split("|")]

    # NEW USER — cluster may have no ratings yet; use global top rated in preferred genres
    if is_new_user(user_id) or len(ch) == 0:
        result = movies_df[movies_df["genre"].str.lower().isin(prefs)]
        result = result.sort_values("avg_rating", ascending=False).head(n).copy()
        result["cluster_avg"] = result["avg_rating"]
        result["rec_reason"]  = "Highest rated in your genre preferences"
        return result

    avg  = ch.groupby("title")["user_rating"].mean().sort_values(ascending=False)
    seen = set(history_df[history_df["user_id"]==user_id]["title"])
    recs = avg[~avg.index.isin(seen)].head(n)
    result = movies_df[movies_df["title"].isin(recs.index)].copy()
    result["cluster_avg"] = result["title"].map(recs)
    return result.sort_values("cluster_avg", ascending=False)

# ── Movie card HTML ───────────────────────────────────────────────────────────
def movie_card(row, score_label="", score_val=""):
    stars = "★" * int(round(row.get("avg_rating", 0)/2)) + "☆" * (5 - int(round(row.get("avg_rating",0)/2)))
    badge_color = {"Action":"#e50914","Drama":"#3b82f6","Comedy":"#f5c518",
                   "Sci-Fi":"#a855f7","Horror":"#1e293b","Romance":"#ec4899",
                   "Thriller":"#f97316","Animation":"#22c55e",
                   "Documentary":"#06b6d4","Fantasy":"#84cc16"}.get(row["genre"],"#7a7a8a")
    score_html = f'<div style="font-size:11px;color:#7a7a8a;margin-top:4px">{score_label} <span style="color:#f5c518;font-weight:600">{score_val}</span></div>' if score_label else ""
    return f"""
    <div style="background:#13151f;border:1px solid #1e2130;border-radius:12px;
                padding:16px;margin-bottom:10px;border-left:3px solid {badge_color};
                transition:all 0.2s">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div style="flex:1">
          <div style="font-weight:600;font-size:15px;color:#e8e6e0;margin-bottom:4px">{row['title']}</div>
          <div style="font-size:12px;color:#7a7a8a;margin-bottom:6px">{row['director']} · {int(row.get('release_year',0))}</div>
          <span style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44;
                       border-radius:20px;padding:2px 10px;font-size:11px;font-weight:500">{row['genre']}</span>
          {score_html}
        </div>
        <div style="text-align:right;margin-left:12px">
          <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#e50914;line-height:1">{row.get('avg_rating','?')}</div>
          <div style="font-size:11px;color:#f5c518">{stars}</div>
          <div style="font-size:10px;color:#7a7a8a;margin-top:2px">{row.get('duration_min','?')} min</div>
        </div>
      </div>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 MyFilms")
    st.markdown("<div style='color:#7a7a8a;font-size:12px;margin-bottom:20px'>AI Movie Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("---")

    # User selector
    st.markdown("### Select User")
    user_options = {f"{r['name']} ({r['user_id']})": r['user_id']
                    for _, r in users_df.iterrows()}
    selected_label = st.selectbox("", list(user_options.keys()), label_visibility="collapsed")
    selected_uid   = user_options[selected_label]

    user_row = users_df[users_df["user_id"] == selected_uid].iloc[0]
    user_hist = history_df[history_df["user_id"] == selected_uid]

    st.markdown("---")
    st.markdown(f"**Age:** {user_row['age']}")
    st.markdown(f"**Plan:** {user_row['subscription_plan']}")
    st.markdown(f"**Cluster:** #{user_row['cluster']}")
    prefs = user_row['preferred_genres'].replace("|", " · ")
    st.markdown(f"**Genres:** {prefs}")
    st.markdown("---")
    st.markdown(f"<div style='color:#7a7a8a;font-size:11px'>100 users · 30 movies · 1054 records</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"# 🎬 {user_row['name']}")
st.markdown(f"<div style='color:#7a7a8a;margin-top:-16px;margin-bottom:20px'>User ID: {selected_uid} · Age {user_row['age']} · {user_row['subscription_plan']} Plan</div>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies Watched",   len(user_hist))
c2.metric("Avg Rating",       f"{user_hist['user_rating'].mean():.1f} / 5")
c3.metric("Completion Rate",  f"{(user_hist['completed']=='Yes').mean()*100:.0f}%")
c4.metric("Fav Genre",        user_hist['genre'].value_counts().index[0] if len(user_hist) else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Profile & History",
    "🎯 Content-Based",
    "🤝 Collaborative",
    "📦 Cluster-Based",
    "📈 Analytics",
    "➕ Add User",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PROFILE & HISTORY
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Genre Distribution")
        genre_counts = user_hist["genre"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(DARK)
        ax.set_facecolor(DARK)
        wedges, texts, autotexts = ax.pie(
            genre_counts.values, labels=genre_counts.index,
            autopct="%1.0f%%", startangle=140,
            colors=COLORS[:len(genre_counts)],
            wedgeprops=dict(width=0.65, edgecolor=DARK, linewidth=2),
        )
        for t in texts: t.set_color(TEXT); t.set_fontsize(9)
        for at in autotexts: at.set_color("#000"); at.set_fontsize(8); at.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("### Ratings Over Time")
        uh_sorted = user_hist.sort_values("watch_date")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(uh_sorted["watch_date"], uh_sorted["user_rating"],
                marker="o", color=RED, linewidth=2, markersize=5,
                markerfacecolor=GOLD, markeredgecolor=RED)
        if len(uh_sorted) >= 3:
            roll = uh_sorted["user_rating"].rolling(3, center=True).mean()
            ax.plot(uh_sorted["watch_date"], roll, color=GOLD,
                    linewidth=1.5, linestyle="--", alpha=0.8, label="3-movie avg")
            ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
        ax.axhline(uh_sorted["user_rating"].mean(), color=MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel("Rating")
        ax.grid(True)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("### Watch History")
    display_hist = user_hist[["title","genre","director","user_rating","watch_date","completed","device"]]\
        .sort_values("watch_date", ascending=False)\
        .rename(columns={"user_rating":"Your Rating","watch_date":"Date",
                          "completed":"Completed","device":"Device"})
    st.dataframe(display_hist, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CONTENT BASED
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Content-Based Recommendations")
    st.markdown("<div style='color:#7a7a8a;font-size:13px;margin-bottom:16px'>Based on movies you rated ★4+ — finds similar films using TF-IDF + cosine similarity on genre & director</div>", unsafe_allow_html=True)

    # New user banner
    if is_new_user(selected_uid):
        st.markdown("""<div style='background:#13151f;border:1px solid #f5c518;border-left:4px solid #f5c518;
            border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#f5c518'>
            ⭐ <b>New user</b> — no watch history yet. Showing top-rated movies in your preferred genres.
            </div>""", unsafe_allow_html=True)

    recs = content_recs(selected_uid, n=6)
    if recs.empty:
        st.info("No recommendations found for the selected genres.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                reason = row.get("rec_reason", "Similarity match")
                st.markdown(movie_card(row, reason, f"{row['avg_rating']}/10"), unsafe_allow_html=True)

    liked = history_df[(history_df["user_id"]==selected_uid) & (history_df["user_rating"]>=4)]["title"].tolist()
    if liked:
        st.markdown("---")
        st.markdown(f"<div style='color:#7a7a8a;font-size:12px'>Driven by your top-rated: <span style='color:#f5c518'>{' · '.join(liked[:5])}</span></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — COLLABORATIVE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Collaborative Filtering Recommendations")
    st.markdown("<div style='color:#7a7a8a;font-size:13px;margin-bottom:16px'>Finds users with similar taste using Pearson correlation, then recommends what they loved</div>", unsafe_allow_html=True)

    # Show similar users
    if selected_uid in rating_matrix.index:
        tv  = rating_matrix.loc[selected_uid].values
        cors = {}
        for uid in rating_matrix.index:
            if uid == selected_uid: continue
            ov   = rating_matrix.loc[uid].values
            mask = (tv!=0) & (ov!=0)
            if mask.sum() < 2: continue
            c, _ = pearsonr(tv[mask], ov[mask])
            if not np.isnan(c): cors[uid] = c
        top3 = sorted(cors.items(), key=lambda x: x[1], reverse=True)[:3]

        sim_cols = st.columns(3)
        for i, (uid, corr) in enumerate(top3):
            uname = users_df[users_df["user_id"]==uid]["name"].values[0]
            with sim_cols[i]:
                st.markdown(f"""<div style='background:#13151f;border:1px solid #1e2130;border-radius:10px;
                    padding:12px;text-align:center;border-top:3px solid {"#e50914" if i==0 else "#f5c518" if i==1 else "#3b82f6"}'>
                    <div style='font-size:11px;color:#7a7a8a'>Similar user #{i+1}</div>
                    <div style='font-weight:600;margin:4px 0'>{uname}</div>
                    <div style='font-size:11px;color:#7a7a8a'>{uid}</div>
                    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;color:#e50914;margin-top:6px'>{corr:.2f}</div>
                    <div style='font-size:10px;color:#7a7a8a'>Pearson r</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # New user banner
    if is_new_user(selected_uid):
        st.markdown("""<div style='background:#13151f;border:1px solid #f5c518;border-left:4px solid #f5c518;
            border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#f5c518'>
            ⭐ <b>New user</b> — no ratings yet. Showing top picks from users who share your genre preferences.
            </div>""", unsafe_allow_html=True)

    recs = collab_recs(selected_uid, n=6)
    if recs.empty:
        st.info("Not enough data to generate collaborative recommendations yet.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                score = f"{row.get('score', 0):.1f}"
                reason = row.get("rec_reason", "Collab score")
                st.markdown(movie_card(row, reason, score), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTER BASED
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    user_cluster = users_df[users_df["user_id"]==selected_uid]["cluster"].values[0]
    cluster_members = users_df[users_df["cluster"]==user_cluster]
    st.markdown(f"### Cluster #{user_cluster} Recommendations")
    st.markdown(f"<div style='color:#7a7a8a;font-size:13px;margin-bottom:16px'>You belong to a group of <span style='color:#f5c518'>{len(cluster_members)} users</span> with similar taste — showing their highest-rated unseen movies</div>", unsafe_allow_html=True)

    # Cluster genre fingerprint
    GENRES = ["Action","Drama","Comedy","Sci-Fi","Horror","Romance","Thriller","Animation"]
    ch     = history_df[history_df["user_id"].isin(cluster_members["user_id"])]
    g_avg  = [ch[ch["genre"]==g]["user_rating"].mean() for g in GENRES]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    bars = ax.barh(GENRES, g_avg, color=COLORS[:len(GENRES)], edgecolor=DARK, linewidth=0.5)
    for bar, val in zip(bars, g_avg):
        if not np.isnan(val):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 5.8)
    ax.set_xlabel("Avg Rating")
    ax.set_title(f"Cluster #{user_cluster} Genre Fingerprint", color=TEXT, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    # New user banner
    if is_new_user(selected_uid):
        st.markdown("""<div style='background:#13151f;border:1px solid #f5c518;border-left:4px solid #f5c518;
            border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#f5c518'>
            ⭐ <b>New user</b> — showing highest rated movies in your preferred genres as a starting point.
            </div>""", unsafe_allow_html=True)

    recs = cluster_recs(selected_uid, n=6)
    if recs.empty:
        st.info("No recommendations available.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                cavg = f"{row.get('cluster_avg', 0):.2f}/5"
                reason = row.get("rec_reason", "Cluster avg rating")
                st.markdown(movie_card(row, reason, cavg), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Platform-Wide Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Most Watched Genres")
        genre_totals = history_df["genre"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(genre_totals.index, genre_totals.values,
                      color=COLORS[:len(genre_totals)], edgecolor=DARK, linewidth=0.5)
        for bar, val in zip(bars, genre_totals.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    str(val), ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_ylabel("Watch Events")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Top Rated Movies")
        top_movies = movies_df.sort_values("avg_rating", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(top_movies["title"], top_movies["avg_rating"],
                       color=RED, edgecolor=DARK, linewidth=0.5)
        for bar, val in zip(bars, top_movies["avg_rating"]):
            ax.text(val + 0.05, bar.get_y()+bar.get_height()/2,
                    f"{val}", va="center", fontsize=8, color=GOLD)
        ax.set_xlim(0, 11)
        ax.set_xlabel("Avg Rating")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("#### Cluster Overview")
    cluster_stats = []
    for c in sorted(users_df["cluster"].unique()):
        members = users_df[users_df["cluster"]==c]["user_id"]
        ch      = history_df[history_df["user_id"].isin(members)]
        top_g   = ch["genre"].value_counts().index[0] if len(ch) else "—"
        cluster_stats.append({
            "Cluster": f"#{c}",
            "Users": len(members),
            "Avg Rating": f"{ch['user_rating'].mean():.2f}",
            "Top Genre": top_g,
            "Total Watches": len(ch),
        })
    st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — ADD USER
# ──────────────────────────────────────────────────────────────────────────────

def save_new_user(name, age, genres, plan):
    users_path   = os.path.join(DATA, "users.csv")
    current      = pd.read_csv(users_path)
    existing_ids = current["user_id"].str.extract(r"U(\d+)")[0].astype(int)
    new_id       = f"U{existing_ids.max() + 1:03d}"
    new_row      = pd.DataFrame([{
        "user_id":           new_id,
        "name":              name,
        "age":               int(age),
        "preferred_genres":  "|".join(genres),
        "subscription_plan": plan,
    }])
    updated = pd.concat([current, new_row], ignore_index=True)
    updated.to_csv(users_path, index=False)
    for excel_path in [
        os.path.join(BASE, "..", "PowerBI_MovieRecommender.xlsx"),
        os.path.join(BASE, "PowerBI_MovieRecommender.xlsx"),
        "/mnt/user-data/outputs/PowerBI_MovieRecommender.xlsx",
    ]:
        if os.path.exists(excel_path):
            try:
                from openpyxl import load_workbook
                wb = load_workbook(excel_path)
                ws = wb["Users"]
                ws.append([new_id, name, int(age), "|".join(genres), plan,
                            0, 0.0, 0.0, genres[0], "—", "—", "Cluster 0"])
                wb.save(excel_path)
            except Exception:
                pass
            break
    load_data.clear()
    build_models.clear()
    return new_id

with tab6:
    GENRES_ALL = ["Action","Drama","Comedy","Sci-Fi","Horror","Romance",
                  "Thriller","Animation","Documentary","Fantasy"]
    PLANS = ["Basic","Standard","Premium"]

    st.markdown("### ➕ Add a New User")
    st.markdown("<div style='color:#7a7a8a;font-size:13px;margin-bottom:20px'>Fill the form and click Save — the new user is written to <b>users.csv</b> and appears in the sidebar immediately.</div>", unsafe_allow_html=True)

    with st.form("add_user_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='color:#f5c518;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:8px'>PERSONAL INFO</div>", unsafe_allow_html=True)
            new_name  = st.text_input("Full Name", placeholder="e.g. John Doe")
            new_age   = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
            new_plan  = st.selectbox("Subscription Plan", PLANS)
        with col2:
            st.markdown("<div style='color:#f5c518;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:8px'>PREFERRED GENRES (1-4)</div>", unsafe_allow_html=True)
            new_genres = st.multiselect("Select genres", GENRES_ALL, default=["Action"], max_selections=4)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✅  Save New User", use_container_width=True)

    if submitted:
        errors = []
        if not new_name.strip():
            errors.append("Name cannot be empty.")
        if not new_genres:
            errors.append("Select at least one genre.")
        current_csv = pd.read_csv(os.path.join(DATA, "users.csv"))
        if new_name.strip() in current_csv["name"].values:
            errors.append(f"A user named '{new_name.strip()}' already exists.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            new_uid = save_new_user(new_name.strip(), int(new_age), new_genres, new_plan)
            st.success(f"✅  {new_name.strip()} added to the dataset with ID {new_uid}!")
            st.markdown(f"""
            <div style='background:#13151f;border:1px solid #22c55e;border-left:4px solid #22c55e;border-radius:12px;padding:16px;margin-top:8px'>
              <div style='font-size:11px;color:#22c55e;font-weight:600;margin-bottom:10px'>NEW USER SAVED TO DATASET</div>
              <div style='display:flex;gap:24px;flex-wrap:wrap'>
                <div><div style='font-size:10px;color:#7a7a8a;text-transform:uppercase'>ID</div><div style='font-size:1.3rem;font-weight:600;color:#e50914'>{new_uid}</div></div>
                <div><div style='font-size:10px;color:#7a7a8a;text-transform:uppercase'>Name</div><div style='font-size:1.3rem;font-weight:600;color:#e8e6e0'>{new_name.strip()}</div></div>
                <div><div style='font-size:10px;color:#7a7a8a;text-transform:uppercase'>Age</div><div style='font-size:1.3rem;font-weight:600;color:#e8e6e0'>{int(new_age)}</div></div>
                <div><div style='font-size:10px;color:#7a7a8a;text-transform:uppercase'>Plan</div><div style='font-size:1.3rem;font-weight:600;color:#f5c518'>{new_plan}</div></div>
                <div><div style='font-size:10px;color:#7a7a8a;text-transform:uppercase'>Genres</div><div style='font-size:1.3rem;font-weight:600;color:#e8e6e0'>{" · ".join(new_genres)}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.rerun()

    st.markdown("---")
    st.markdown("### 👥 All Users in Dataset")
    live_users = pd.read_csv(os.path.join(DATA, "users.csv"))
    live_users = live_users[["user_id","name","age","preferred_genres","subscription_plan"]]
    live_users.columns = ["ID","Name","Age","Preferred Genres","Plan"]
    st.dataframe(live_users, use_container_width=True, hide_index=True)
    st.markdown(f"<div style='color:#7a7a8a;font-size:11px;margin-top:4px'>{len(live_users)} users in dataset</div>", unsafe_allow_html=True)
