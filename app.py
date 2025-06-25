import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta, datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Utility functions
@st.cache_data(show_spinner=False)
def load_sentiment():
    sentiment = pd.read_csv("fear_greed_index.csv", parse_dates=["Date"])
    sentiment["Date"] = sentiment["Date"].dt.date
    return sentiment

@st.cache_data(show_spinner=False)
def load_default_data():
    df = pd.read_csv("data/merged_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    summary = pd.read_csv("data/daily_summary.csv")
    summary["Date"] = pd.to_datetime(summary["Date"])
    return df, summary

# Sidebar UI
st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=120)
st.sidebar.title("📈 Bitcoin Trading Analyzer")
st.sidebar.markdown("Analyze performance vs. market sentiment (Fear/Greed).")

# Upload functionality
st.sidebar.subheader("📤 Upload Trade Data")
uploaded_file = st.sidebar.file_uploader("Choose a trade CSV", type="csv")
sentiment_data = load_sentiment()

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"data/uploads/trades_{timestamp}.csv"
    os.makedirs("data/uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(temp_path, parse_dates=["time"])
    df["Date"] = df["time"].dt.date
    df = pd.merge(df, sentiment_data, on="Date", how="inner")
    df["Date"] = pd.to_datetime(df["Date"])
    df["sentiment_flag"] = df["Classification"].map({"Fear": 0, "Greed": 1, "Neutral": 0.5})
    st.success("✅ Data uploaded and merged with sentiment.")
    summary = df.groupby(["Date", "Classification"]).agg({
        "closedPnL": ["mean", "sum"],
        "leverage": "mean",
        "size": "sum",
        "execution price": "mean"
    }).reset_index()
    summary.columns = ['Date', 'Classification', 'AvgPnL', 'TotalPnL', 'AvgLeverage', 'TotalSize', 'AvgPrice']
else:
    df, summary = load_default_data()

# Date filter
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
default_start = max_date - timedelta(days=30)

st.sidebar.subheader("📅 Date Range")
date_range = st.sidebar.date_input("Select range:", [default_start, max_date])

if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    summary = summary[(summary["Date"] >= start) & (summary["Date"] <= end)]

    if df.empty:
        st.warning("⚠️ No data in this range.")
        st.stop()

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Trades", len(df))
with col2:
    st.metric("💰 PnL", f"${df['closedPnL'].sum():,.2f}")
with col3:
    win_rate = (df['closedPnL'] > 0).mean() * 100
    st.metric("🏆 Win %", f"{win_rate:.1f}%")

# Win rate by sentiment
st.subheader("🎯 Win Rate by Sentiment")
df['win'] = df['closedPnL'] > 0
sentiment_win_rate = df.groupby("Classification")["win"].mean().reset_index()
fig1 = px.pie(sentiment_win_rate, names="Classification", values="win", hole=0.4)
st.plotly_chart(fig1, use_container_width=True)

# PnL time series
st.subheader("📈 Total PnL Over Time")
fig2 = px.line(summary, x="Date", y="TotalPnL", color="Classification")
st.plotly_chart(fig2, use_container_width=True)

# Boxplot
st.subheader("📦 PnL Distribution")
fig3 = px.box(df, x="Classification", y="closedPnL", color="Classification")
st.plotly_chart(fig3, use_container_width=True)

# High leverage
st.subheader("⚠️ High Leverage Trades")
high_lev = df[df["leverage"] > 10]
fig4 = px.scatter(high_lev, x="leverage", y="closedPnL", color="Classification", hover_data=["account"])
st.plotly_chart(fig4, use_container_width=True)

# Trader breakdown
st.subheader("🧠 Trader Insights")
traders = df["account"].unique()
selected_trader = st.selectbox("Choose trader", traders)
trader_df = df[df["account"] == selected_trader]
fig_trader = px.bar(trader_df, x="Date", y="closedPnL", color="Classification")
st.plotly_chart(fig_trader, use_container_width=True)

# Download
st.download_button("📥 Download CSV", df.to_csv(index=False), "filtered_data.csv")

# ML prediction
st.markdown("---")
st.subheader("🤖 Trade Success Prediction")
try:
    df["target"] = (df["closedPnL"] > 0).astype(int)
    X = df[["leverage", "size", "sentiment_flag"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.info(f"Logistic Regression Accuracy: {acc*100:.2f}%")
except Exception as e:
    st.error(f"ML Error: {e}")

# Chatbot-like Q&A
st.markdown("---")
st.subheader("💬 Ask a question")
query = st.chat_input("Try: best trader, worst trader, most profitable day")
if query:
    st.write(f"You asked: {query}")
    q = query.lower()
    if "best trader" in q:
        best = df.groupby("account")["closedPnL"].sum().idxmax()
        st.success(f"📈 Best: {best}")
    elif "worst trader" in q:
        worst = df.groupby("account")["closedPnL"].sum().idxmin()
        st.error(f"📉 Worst: {worst}")
    elif "profitable day" in q:
        best_day = summary.groupby("Date")["TotalPnL"].sum().idxmax()
        st.success(f"💹 Most Profitable Day: {best_day.date()}")
    else:
        st.info("Try 'best trader', 'worst trader', or 'most profitable day'")
