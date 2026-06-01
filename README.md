# Bitcoin Sentiment vs Trader Performance
### Primetrade.ai — Data Science Assignment

---

## What This Project Does

We have two datasets:
- **fear_greed_index.csv** — daily Bitcoin Fear & Greed scores (2018–2025)
- **historical_data.csv** — 211k+ real trades from Hyperliquid DEX (32 traders, 246 coins)

The goal: find out whether the **market mood** (fearful or greedy) actually affects
**how well traders perform**, and surface any hidden patterns that could help
build smarter trading strategies.

---

## Project Structure

```
bitcoin_sentiment_analysis/
│
├── analysis.py                  ← main script, run this
├── fear_greed_index.csv         ← dataset 1 (sentiment)
├── historical_data.csv          ← dataset 2 (trades)
├── requirements.txt             ← pip dependencies
│
└── outputs/                     ← all charts + summary get saved here
    ├── 01_trade_count_by_sentiment.png
    ├── 02_avg_pnl_by_sentiment.png
    ├── 03_pnl_distribution_boxplot.png
    ├── 04_win_rate_by_sentiment.png
    ├── 05_avg_trade_size_by_sentiment.png
    ├── 06_hourly_activity_pnl.png
    ├── 07_top_coins_pnl.png
    ├── 08_long_vs_short_by_sentiment.png
    ├── 09_sentiment_pnl_heatmap.png
    ├── 10_top_accounts_pnl.png
    └── summary_by_sentiment.csv
```

---

## How to Run

### Step 1 — Clone / download the project
If you're using Git:
```bash
git clone <your-repo-url>
cd bitcoin_sentiment_analysis
```
Or just put all files in one folder and open that folder in VS Code.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the analysis
```bash
python analysis.py
```
That's it. All 10 charts + 1 summary CSV will appear inside the `outputs/` folder.

---

## Key Findings

| Sentiment     | Avg PnL (USD) | Win Rate | Trade Count |
|---------------|--------------|----------|-------------|
| Extreme Fear  | $71           | 76.2%    | 10,406      |
| Fear          | $113          | 87.3%    | 29,808      |
| Neutral       | $71           | 82.4%    | 18,159      |
| Greed         | $85           | 76.9%    | 25,176      |
| Extreme Greed | $130          | 89.2%    | 20,853      |

**Top Insights:**
1. **Extreme Greed = best average PnL** ($130 per closing trade)
2. **Extreme Greed also has the highest win rate** at 89.2%
3. **Fear regime is the most active** — traders place the most trades when the market is fearful
4. **Trade sizes are smaller during Extreme Greed**, meaning traders are more disciplined, not just lucky
5. **Long positions outperform Short positions** in every sentiment regime
6. Certain hours of the day show significantly higher PnL — timing matters

---

## Charts Explained

| Chart | What it shows |
|-------|---------------|
| 01    | How many trades happen in each sentiment environment |
| 02    | Average profit per trade by sentiment |
| 03    | PnL spread (box plot) — shows how consistent results are |
| 04    | What % of trades are winners in each sentiment |
| 05    | How much money traders risk per trade in each sentiment |
| 06    | Which hours of the day are busiest and most profitable |
| 07    | Which coins make (or lose) the most total money |
| 08    | Longs vs Shorts performance side by side |
| 09    | Heatmap of sentiment score vs PnL magnitude |
| 10    | Which trader accounts are most profitable overall |

---

## Dependencies

See `requirements.txt`. All standard data science libraries, nothing exotic.
