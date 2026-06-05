"""
Visualizations
- Pie chart: genre distribution in a user's watch history
- Line plot: user ratings over time
- Bar chart: platform-wide genre trends
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

DARK_BG   = "#0F1117"
CARD_BG   = "#1A1D27"
RED       = "#E50914"
GOLD      = "#F5C518"
TEXT_COL  = "#E8E8E8"
MUTED     = "#9A9A9A"
PALETTE   = ["#E50914","#F5C518","#3B82F6","#22C55E","#A855F7",
             "#F97316","#06B6D4","#EC4899","#84CC16","#EAB308"]

def _apply_dark(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)


def plot_genre_pie(user_id: str, history_df: pd.DataFrame,
                   name: str = "", output_dir: str = "outputs"):
    user_hist    = history_df[history_df["user_id"] == user_id]
    genre_counts = user_hist["genre"].value_counts()
    label        = name or user_id

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    wedges, texts, autotexts = ax.pie(
        genre_counts.values,
        labels=genre_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=PALETTE[:len(genre_counts)],
        wedgeprops=dict(width=0.65, edgecolor=DARK_BG, linewidth=2),
    )
    for t in texts:
        t.set_color(TEXT_COL)
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color("#000000")
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.set_title(f"Genre Distribution — {label}", color=TEXT_COL,
                 fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    path = os.path.join(output_dir, f"genre_pie_{user_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  📊 Saved: {path}")
    return path


def plot_ratings_over_time(user_id: str, history_df: pd.DataFrame,
                            name: str = "", output_dir: str = "outputs"):
    user_hist = history_df[history_df["user_id"] == user_id].copy()
    user_hist["watch_date"] = pd.to_datetime(user_hist["watch_date"])
    user_hist = user_hist.sort_values("watch_date")
    label     = name or user_id

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark(fig, ax)

    ax.plot(user_hist["watch_date"], user_hist["user_rating"],
            marker="o", color=RED, linewidth=2, markersize=6,
            markerfacecolor=GOLD, markeredgecolor=RED)

    # Rolling average
    if len(user_hist) >= 3:
        user_hist["rolling"] = user_hist["user_rating"].rolling(3, center=True).mean()
        ax.plot(user_hist["watch_date"], user_hist["rolling"],
                color=GOLD, linewidth=1.5, linestyle="--", alpha=0.7, label="3-movie avg")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=9)

    ax.set_title(f"Ratings Over Time — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating (1–5)")
    ax.set_ylim(0.5, 5.5)
    ax.axhline(user_hist["user_rating"].mean(), color=MUTED, linestyle=":", linewidth=1)
    ax.grid(color=MUTED, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(output_dir, f"ratings_time_{user_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  📈 Saved: {path}")
    return path


def plot_platform_genres(history_df: pd.DataFrame, output_dir: str = "outputs"):
    genre_totals = history_df["genre"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark(fig, ax)

    bars = ax.bar(genre_totals.index, genre_totals.values,
                  color=PALETTE[:len(genre_totals)], edgecolor=DARK_BG, linewidth=1.2)

    for bar, val in zip(bars, genre_totals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", color=TEXT_COL, fontsize=8)

    ax.set_title("Most-Watched Genres — Platform Wide", fontsize=13, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Watch Events")
    plt.xticks(rotation=25, ha="right", color=TEXT_COL)
    plt.tight_layout()
    path = os.path.join(output_dir, "platform_genres.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  📉 Saved: {path}")
    return path


def plot_cluster_heatmap(users_df: pd.DataFrame, history_df: pd.DataFrame,
                          output_dir: str = "outputs"):
    GENRES = ["Action","Drama","Comedy","Sci-Fi","Horror",
              "Romance","Thriller","Animation","Documentary","Fantasy"]

    cluster_genre_avg = {}
    for cluster in sorted(users_df["cluster"].unique()):
        members = users_df[users_df["cluster"] == cluster]["user_id"]
        ch      = history_df[history_df["user_id"].isin(members)]
        row     = {g: ch[ch["genre"]==g]["user_rating"].mean() for g in GENRES}
        cluster_genre_avg[f"Cluster {cluster}"] = row

    df_heat = pd.DataFrame(cluster_genre_avg).T.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    sns.heatmap(df_heat, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, linecolor=DARK_BG,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Average Genre Ratings per Cluster", color=TEXT_COL,
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors=TEXT_COL)
    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  🌡️  Saved: {path}")
    return path
