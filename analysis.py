# ============================================================
#  Bitcoin Sentiment vs Trader Performance — Full Analysis
#  Primetrade.ai Data Science Assignment
# ============================================================
# What we're doing here:
#   1. Load and clean both datasets
#   2. Merge them on date so every trade has a sentiment label
#   3. Explore how traders behave differently under Fear vs Greed
#   4. Find which sentiment regime is most profitable
#   5. Uncover hidden patterns (best coins, best traders, risky hours)
#   6. Save charts and a summary table for the final report
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── make sure output folder exists ──────────────────────────
os.makedirs("outputs", exist_ok=True)

# set a clean style for all plots
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 120


# ════════════════════════════════════════════════════════════
# SECTION 1 — Load the data
# ════════════════════════════════════════════════════════════

print("Loading datasets...")

# Fear & Greed Index — daily sentiment scores for Bitcoin
fear_greed = pd.read_csv("fear_greed_index.csv")

# Historical trades pulled from Hyperliquid (a crypto DEX)
trades = pd.read_csv("historical_data.csv")

print(f"  Fear/Greed rows : {len(fear_greed):,}")
print(f"  Trades rows     : {len(trades):,}")


# ════════════════════════════════════════════════════════════
# SECTION 2 — Clean & parse dates
# ════════════════════════════════════════════════════════════

print("\nCleaning data...")

# -- Fear & Greed --
# The 'date' column is already a string like '2018-02-01', just parse it
fear_greed["date"] = pd.to_datetime(fear_greed["date"])

# -- Trades --
# Timestamp IST looks like '02-12-2024 22:50' → day-month-year format
trades["trade_date"] = pd.to_datetime(
    trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
)

# Drop rows where date parsing failed (corrupted timestamps)
bad_dates = trades["trade_date"].isna().sum()
if bad_dates > 0:
    print(f"  Dropping {bad_dates} rows with unparseable timestamps")
    trades = trades.dropna(subset=["trade_date"])

# Extract just the calendar date (no time) so we can join on it
trades["date_only"] = trades["trade_date"].dt.date
fear_greed["date_only"] = fear_greed["date"].dt.date

# Also grab hour for intraday pattern analysis later
trades["hour"] = trades["trade_date"].dt.hour

# Rename a few columns to make them easier to work with
trades = trades.rename(columns={
    "Closed PnL":        "closed_pnl",
    "Size USD":          "size_usd",
    "Execution Price":   "exec_price",
    "Size Tokens":       "size_tokens",
    "Account":           "account",
    "Coin":              "coin",
    "Side":              "side",
    "Direction":         "direction",
    "Fee":               "fee",
})

print("  Cleaning done.")


# ════════════════════════════════════════════════════════════
# SECTION 3 — Merge trades with sentiment
# ════════════════════════════════════════════════════════════

# We join on the calendar date so every trade inherits
# that day's sentiment classification (Fear, Greed, etc.)
merged = trades.merge(
    fear_greed[["date_only", "value", "classification"]],
    on="date_only",
    how="left"
)

# Rename to avoid confusion
merged = merged.rename(columns={
    "value":          "sentiment_score",
    "classification": "sentiment"
})

# How many trades didn't get a sentiment? (dates outside F/G range)
no_sentiment = merged["sentiment"].isna().sum()
print(f"\nTrades without a matching sentiment date: {no_sentiment:,}")
merged = merged.dropna(subset=["sentiment"])
print(f"Trades kept for analysis: {len(merged):,}")

# Define a clean ordering for sentiment categories
sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
merged["sentiment"] = pd.Categorical(
    merged["sentiment"], categories=sentiment_order, ordered=True
)


# ════════════════════════════════════════════════════════════
# SECTION 4 — Basic Stats
# ════════════════════════════════════════════════════════════

print("\n── BASIC STATS ──────────────────────────────────────────")

# How many trades fall into each sentiment bucket?
trade_counts = merged["sentiment"].value_counts().reindex(sentiment_order)
print("\nTrade count by sentiment:")
print(trade_counts.to_string())

# Average PnL per sentiment (only closing trades have non-zero PnL)
closing_trades = merged[merged["closed_pnl"] != 0].copy()
print(f"\nClosing trades (where PnL is recorded): {len(closing_trades):,}")

avg_pnl = closing_trades.groupby("sentiment", observed=True)["closed_pnl"].mean()
print("\nAverage Closed PnL by Sentiment:")
print(avg_pnl.round(2).to_string())


# ════════════════════════════════════════════════════════════
# SECTION 5 — Plots
# ════════════════════════════════════════════════════════════

# ── 5A. Trade Volume by Sentiment ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#d73027", "#f46d43", "#fdae61", "#74add1", "#313695"]
trade_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
ax.set_title("Number of Trades in Each Sentiment Regime", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Trade Count", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/01_trade_count_by_sentiment.png")
plt.close()
print("\nSaved: outputs/01_trade_count_by_sentiment.png")

# ── 5B. Average PnL by Sentiment ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["green" if v > 0 else "crimson" for v in avg_pnl.values]
avg_pnl.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white", width=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Average Closed PnL by Sentiment Regime", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Avg Closed PnL (USD)", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("outputs/02_avg_pnl_by_sentiment.png")
plt.close()
print("Saved: outputs/02_avg_pnl_by_sentiment.png")

# ── 5C. PnL Distribution (box plot) ─────────────────────────
# clip extreme outliers so the chart is readable
clip_val = closing_trades["closed_pnl"].quantile(0.99)
clipped = closing_trades[closing_trades["closed_pnl"].abs() < clip_val]

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=clipped,
    x="sentiment", y="closed_pnl",
    order=sentiment_order,
    palette=["#d73027", "#f46d43", "#fdae61", "#74add1", "#313695"],
    ax=ax, linewidth=0.8
)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("PnL Distribution per Sentiment (99th-percentile clipped)", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Closed PnL (USD)", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("outputs/03_pnl_distribution_boxplot.png")
plt.close()
print("Saved: outputs/03_pnl_distribution_boxplot.png")

# ── 5D. Win Rate by Sentiment ────────────────────────────────
# A "win" = closed PnL > 0
closing_trades["is_win"] = closing_trades["closed_pnl"] > 0
win_rate = closing_trades.groupby("sentiment", observed=True)["is_win"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
win_rate.plot(kind="bar", ax=ax, color="#2196F3", edgecolor="white", width=0.6)
ax.axhline(50, color="red", linewidth=1, linestyle="--", label="50% breakeven")
ax.set_title("Win Rate (%) by Sentiment Regime", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Win Rate (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend()
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("outputs/04_win_rate_by_sentiment.png")
plt.close()
print("Saved: outputs/04_win_rate_by_sentiment.png")

# ── 5E. Trade Size (USD) by Sentiment ───────────────────────
avg_size = merged.groupby("sentiment", observed=True)["size_usd"].mean()

fig, ax = plt.subplots(figsize=(9, 5))
avg_size.plot(kind="bar", ax=ax, color="#9C27B0", edgecolor="white", width=0.6)
ax.set_title("Average Trade Size (USD) by Sentiment", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Avg Trade Size (USD)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("outputs/05_avg_trade_size_by_sentiment.png")
plt.close()
print("Saved: outputs/05_avg_trade_size_by_sentiment.png")

# ── 5F. Hourly Trading Activity ─────────────────────────────
hourly = merged.groupby("hour")["closed_pnl"].agg(["count", "mean"]).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.bar(hourly["hour"], hourly["count"], color="#90CAF9", label="Trade Count", alpha=0.7)
ax2.plot(hourly["hour"], hourly["mean"], color="#E53935", marker="o", linewidth=2, label="Avg PnL")
ax1.set_xlabel("Hour of Day (IST)", fontsize=11)
ax1.set_ylabel("Trade Count", fontsize=11)
ax2.set_ylabel("Avg Closed PnL (USD)", fontsize=11)
ax1.set_title("Hourly Trading Activity & Average PnL", fontsize=13, pad=12)
ax1.set_xticks(range(24))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/06_hourly_activity_pnl.png")
plt.close()
print("Saved: outputs/06_hourly_activity_pnl.png")

# ── 5G. Top 10 Most Traded Coins ─────────────────────────────
top_coins = (
    closing_trades.groupby("coin")["closed_pnl"]
    .agg(total_pnl="sum", trades="count")
    .sort_values("trades", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
top_coins["total_pnl"].plot(kind="bar", ax=ax, color="#FF9800", edgecolor="white", width=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Total PnL for Top-10 Most Traded Coins", fontsize=13, pad=12)
ax.set_xlabel("Coin", fontsize=11)
ax.set_ylabel("Total Closed PnL (USD)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("outputs/07_top_coins_pnl.png")
plt.close()
print("Saved: outputs/07_top_coins_pnl.png")

# ── 5H. Long vs Short Performance by Sentiment ───────────────
# Only look at directional trades (not spot/conversion rows)
directional = closing_trades[
    closing_trades["direction"].isin(["Close Long", "Close Short"])
].copy()
directional["trade_type"] = directional["direction"].map(
    {"Close Long": "Long", "Close Short": "Short"}
)

long_short_pnl = (
    directional.groupby(["sentiment", "trade_type"], observed=True)["closed_pnl"]
    .mean()
    .unstack("trade_type")
)

fig, ax = plt.subplots(figsize=(10, 6))
long_short_pnl.plot(kind="bar", ax=ax, edgecolor="white", width=0.6,
                     color={"Long": "#43A047", "Short": "#E53935"})
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Avg PnL: Long vs Short Positions by Sentiment", fontsize=13, pad=12)
ax.set_xlabel("Market Sentiment", fontsize=11)
ax.set_ylabel("Avg Closed PnL (USD)", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(title="Position Type")
plt.tight_layout()
plt.savefig("outputs/08_long_vs_short_by_sentiment.png")
plt.close()
print("Saved: outputs/08_long_vs_short_by_sentiment.png")

# ── 5I. Heatmap — Sentiment Score vs Trader PnL ─────────────
# Bin sentiment score into deciles and cross-tab with PnL buckets
closing_trades["score_bin"] = pd.cut(
    closing_trades["sentiment_score"], bins=5,
    labels=["0-20", "21-40", "41-60", "61-80", "81-100"]
)
closing_trades["pnl_bin"] = pd.cut(
    closing_trades["closed_pnl"].clip(-500, 500), bins=10
)
heat_data = (
    closing_trades.groupby(["score_bin", "pnl_bin"], observed=True)
    .size()
    .unstack("pnl_bin")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(heat_data, cmap="YlOrRd", ax=ax, linewidths=0.3, cbar_kws={"label": "Trade Count"})
ax.set_title("Heatmap: Sentiment Score Bucket vs PnL Bucket", fontsize=13, pad=12)
ax.set_xlabel("Closed PnL Range (USD, clipped at ±500)", fontsize=10)
ax.set_ylabel("Sentiment Score Range", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/09_sentiment_pnl_heatmap.png")
plt.close()
print("Saved: outputs/09_sentiment_pnl_heatmap.png")

# ── 5J. Per-Account Performance ──────────────────────────────
account_stats = (
    closing_trades.groupby("account")["closed_pnl"]
    .agg(total_pnl="sum", avg_pnl="mean", trade_count="count", win_rate=lambda x: (x > 0).mean() * 100)
    .sort_values("total_pnl", ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    account_stats.index.str[:10] + "…",
    account_stats["total_pnl"],
    color=["#43A047" if v > 0 else "#E53935" for v in account_stats["total_pnl"]],
    edgecolor="white"
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top-15 Accounts by Total Closed PnL", fontsize=13, pad=12)
ax.set_xlabel("Total Closed PnL (USD)", fontsize=11)
ax.set_ylabel("Account (truncated)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/10_top_accounts_pnl.png")
plt.close()
print("Saved: outputs/10_top_accounts_pnl.png")


# ════════════════════════════════════════════════════════════
# SECTION 6 — Summary Table (saved to CSV)
# ════════════════════════════════════════════════════════════

print("\n── BUILDING SUMMARY TABLE ───────────────────────────────")

summary = closing_trades.groupby("sentiment", observed=True).agg(
    total_trades    = ("closed_pnl", "count"),
    total_pnl       = ("closed_pnl", "sum"),
    avg_pnl         = ("closed_pnl", "mean"),
    median_pnl      = ("closed_pnl", "median"),
    std_pnl         = ("closed_pnl", "std"),
    win_rate_pct    = ("is_win",     lambda x: round(x.mean() * 100, 1)),
    avg_trade_size  = ("size_usd",   "mean"),
).reset_index()

summary = summary.sort_values("sentiment")
summary.to_csv("outputs/summary_by_sentiment.csv", index=False)
print("Saved: outputs/summary_by_sentiment.csv")
print()
print(summary.to_string(index=False))


# ════════════════════════════════════════════════════════════
# SECTION 7 — Key Insights (printed to console)
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

best_sentiment = summary.loc[summary["avg_pnl"].idxmax(), "sentiment"]
worst_sentiment = summary.loc[summary["avg_pnl"].idxmin(), "sentiment"]
highest_vol = summary.loc[summary["total_trades"].idxmax(), "sentiment"]

print(f"\n1. Best avg PnL sentiment  : {best_sentiment}")
print(f"2. Worst avg PnL sentiment : {worst_sentiment}")
print(f"3. Most active regime      : {highest_vol}")
print(f"4. Overall win rate        : {closing_trades['is_win'].mean()*100:.1f}%")
print(f"5. Unique traders analysed : {merged['account'].nunique()}")
print(f"6. Unique coins traded     : {merged['coin'].nunique()}")
print(f"7. Date range of trades    : {merged['trade_date'].min().date()} → {merged['trade_date'].max().date()}")

print("\nAll charts saved to the 'outputs/' folder.")
print("Done ✓")
