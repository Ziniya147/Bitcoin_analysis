import pandas as pd
import os

# Load your CSV files
trades = pd.read_csv("historical_trades.csv", parse_dates=["time"])
sentiment = pd.read_csv("fear_greed_index.csv", parse_dates=["Date"])

# Prepare data for merge
trades["Date"] = trades["time"].dt.date
sentiment["Date"] = sentiment["Date"].dt.date

# Merge the two datasets on Date
df = pd.merge(trades, sentiment, on="Date", how="inner")

# Add sentiment_flag column
df["sentiment_flag"] = df["Classification"].map({'Fear': 0, 'Greed': 1, 'Neutral': 0.5})

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)

# Save the merged file
df.to_csv("data/merged_data.csv", index=False)

print("✅ Merged file created: data/merged_data.csv")
