import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# File paths
RAW_TRADES_PATH = "historical_trades.csv"
RAW_SENTIMENT_PATH = "fear_greed_index.csv"
MERGED_OUTPUT_PATH = "data/merged_data.csv"
SUMMARY_OUTPUT_PATH = "data/daily_summary.csv"
EDA_PLOT_PATH = "data/pnl_boxplot.png"

# Ensure output directories
os.makedirs("data", exist_ok=True)


def load_data(trades_path=RAW_TRADES_PATH, sentiment_path=RAW_SENTIMENT_PATH):
    trades = pd.read_csv(trades_path, parse_dates=["time"])
    sentiment = pd.read_csv(sentiment_path, parse_dates=["Date"])
    return trades, sentiment


def preprocess_and_merge(trades, sentiment):
    trades['Date'] = trades['time'].dt.date
    sentiment['Date'] = sentiment['Date'].dt.date
    df = pd.merge(trades, sentiment, on="Date", how="inner")
    df['sentiment_flag'] = df['Classification'].map({'Fear': 0, 'Greed': 1, 'Neutral': 0.5})
    return df


def generate_summary(df):
    summary = df.groupby(['Date', 'Classification']).agg({
        'closedPnL': ['mean', 'sum'],
        'leverage': 'mean',
        'size': 'sum',
        'execution price': 'mean'
    }).reset_index()
    summary.columns = ['Date', 'Classification', 'AvgPnL', 'TotalPnL', 'AvgLeverage', 'TotalSize', 'AvgPrice']
    return summary


def save_outputs(df, summary):
    df.to_csv(MERGED_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"✅ Merged data saved to {MERGED_OUTPUT_PATH}")
    print(f"✅ Summary saved to {SUMMARY_OUTPUT_PATH}")


def generate_eda_plot(df, output_path=EDA_PLOT_PATH):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Classification', y='closedPnL')
    plt.title("Trader PnL Distribution by Market Sentiment")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 EDA plot saved to {output_path}")


def main():
    print("🔁 Starting data processing...")
    trades, sentiment = load_data()
    df = preprocess_and_merge(trades, sentiment)
    summary = generate_summary(df)
    save_outputs(df, summary)
    generate_eda_plot(df)
    print("✅ Data processing complete.")


if __name__ == "__main__":
    main()
