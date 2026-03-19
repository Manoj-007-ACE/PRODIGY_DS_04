import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import re

# ============================================================
# Prodigy InfoTech - Data Science Internship
# Task 4: Sentiment Analysis - Twitter Dataset
# ============================================================

df = pd.read_csv(r"C:\prodigy_task4\twitter_training.csv",
                 header=None,
                 names=["id", "topic", "sentiment", "text"])

# ==================== DATA CLEANING ====================
df.dropna(subset=["text"], inplace=True)
df = df[df["sentiment"] != "Irrelevant"]
df["text"] = df["text"].astype(str)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning Done! Shape:", df.shape)
print("\nSentiment counts:\n", df["sentiment"].value_counts())

sns.set_style("whitegrid")
colors = {"Positive": "#1D9E75", "Negative": "#E24B4A", "Neutral": "#378ADD"}

# ==================== PLOT 1: Sentiment Distribution ====================
plt.figure(figsize=(8, 5))
order = ["Positive", "Negative", "Neutral"]
palette = [colors[s] for s in order]
ax = sns.countplot(x="sentiment", data=df, order=order, palette=palette)
ax.bar_label(ax.containers[0], fontsize=12, fontweight="bold")
plt.title("Overall Sentiment Distribution", fontsize=15, fontweight="bold")
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Number of Tweets", fontsize=12)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot1_sentiment_distribution.png", dpi=150)
plt.show()
print("Plot 1 saved!")

# ==================== PLOT 2: Sentiment by Top 10 Topics ====================
top10_topics = df["topic"].value_counts().head(10).index
df_top10 = df[df["topic"].isin(top10_topics)]

plt.figure(figsize=(14, 6))
topic_sentiment = df_top10.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
topic_sentiment = topic_sentiment[["Positive", "Negative", "Neutral"]]
topic_sentiment.plot(kind="bar", figsize=(14, 6),
                     color=["#1D9E75", "#E24B4A", "#378ADD"],
                     edgecolor="white")
plt.title("Sentiment by Top 10 Topics", fontsize=15, fontweight="bold")
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Number of Tweets", fontsize=12)
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.legend(["Positive", "Negative", "Neutral"], fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot2_sentiment_by_topic.png", dpi=150)
plt.show()
print("Plot 2 saved!")

# ==================== PLOT 3: Sentiment Pie Chart ====================
plt.figure(figsize=(7, 7))
sentiment_counts = df["sentiment"].value_counts()
explode = (0.05, 0.05, 0.05)
plt.pie(sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=["#E24B4A", "#1D9E75", "#378ADD"],
        explode=explode,
        startangle=140,
        textprops={"fontsize": 13})
plt.title("Sentiment Share (Pie Chart)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot3_sentiment_pie.png", dpi=150)
plt.show()
print("Plot 3 saved!")

# ==================== PLOT 4: WordCloud - Positive Tweets ====================
positive_text = " ".join(df[df["sentiment"] == "Positive"]["clean_text"])
wordcloud_pos = WordCloud(width=1200, height=600,
                          background_color="white",
                          colormap="Greens",
                          max_words=150).generate(positive_text)
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Positive Tweets", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot4_wordcloud_positive.png", dpi=150)
plt.show()
print("Plot 4 saved!")

# ==================== PLOT 5: WordCloud - Negative Tweets ====================
negative_text = " ".join(df[df["sentiment"] == "Negative"]["clean_text"])
wordcloud_neg = WordCloud(width=1200, height=600,
                          background_color="white",
                          colormap="Reds",
                          max_words=150).generate(negative_text)
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Tweets", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot5_wordcloud_negative.png", dpi=150)
plt.show()
print("Plot 5 saved!")

# ==================== PLOT 6: Top 10 Most Positive vs Negative Topics ====================
topic_sentiment_pct = df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
topic_sentiment_pct["total"] = topic_sentiment_pct.sum(axis=1)
topic_sentiment_pct["pos_pct"] = topic_sentiment_pct["Positive"] / topic_sentiment_pct["total"] * 100
topic_sentiment_pct["neg_pct"] = topic_sentiment_pct["Negative"] / topic_sentiment_pct["total"] * 100

top5_pos = topic_sentiment_pct.nlargest(5, "pos_pct")[["pos_pct"]]
top5_neg = topic_sentiment_pct.nlargest(5, "neg_pct")[["neg_pct"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

top5_pos.plot(kind="barh", ax=ax1, color="#1D9E75", edgecolor="white", legend=False)
ax1.set_title("Top 5 Most Positive Topics", fontsize=13, fontweight="bold")
ax1.set_xlabel("Positive Tweet %", fontsize=11)
ax1.invert_yaxis()

top5_neg.plot(kind="barh", ax=ax2, color="#E24B4A", edgecolor="white", legend=False)
ax2.set_title("Top 5 Most Negative Topics", fontsize=13, fontweight="bold")
ax2.set_xlabel("Negative Tweet %", fontsize=11)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(r"C:\prodigy_task4\plot6_pos_neg_topics.png", dpi=150)
plt.show()
print("Plot 6 saved!")

print("\nAll 6 plots saved in C:\\prodigy_task4\\")
print("\nKey Insights:")
print(f"Total tweets analysed: {len(df):,}")
print(f"Positive: {len(df[df['sentiment']=='Positive']):,} ({len(df[df['sentiment']=='Positive'])/len(df)*100:.1f}%)")
print(f"Negative: {len(df[df['sentiment']=='Negative']):,} ({len(df[df['sentiment']=='Negative'])/len(df)*100:.1f}%)")
print(f"Neutral:  {len(df[df['sentiment']=='Neutral']):,} ({len(df[df['sentiment']=='Neutral'])/len(df)*100:.1f}%)")