PRODIGY_DS_04 — Twitter Sentiment Analysis

## 📌 Task
Analyze and visualize sentiment patterns in social 
media data to understand public opinion and attitudes 
towards specific topics or brands.

## 🛠 Tools Used
Python | Pandas | Matplotlib | Seaborn | NLTK | WordCloud | NumPy

## 🧹 Data Cleaning Steps
- Removed irrelevant tweets from dataset
- Dropped rows with missing tweet text (686 missing)
- Removed URLs, mentions (@user), hashtags (#tag)
- Removed special characters and numbers
- Converted all text to lowercase

## 📊 Dataset Details
- Total tweets analyzed: 61,692
- Topics/Brands covered: 20+
- Sentiments: Positive, Negative, Neutral
- Source: Twitter Entity Sentiment Dataset (Kaggle)

## 🎯 Key Findings
- Negative: 36.6% — people complain more on Twitter!
- Positive: 33.8%
- Neutral: 29.6%
- Most positive words: love, game, thank, great, amazing
- Most negative words: fix, cant, broken, worst, hate
- MaddenNFL = most negatively discussed brand
- AssassinsCreed = most positively discussed brand

## 📈 Visualizations Created
- plot1_sentiment_distribution.png — Overall sentiment bar chart
- plot2_sentiment_by_topic.png — Sentiment by top 10 topics
- plot3_sentiment_pie.png — Sentiment share pie chart
- plot4_wordcloud_positive.png — Word cloud positive tweets
- plot5_wordcloud_negative.png — Word cloud negative tweets
- plot6_pos_neg_topics.png — Most positive vs negative topics

## 📁 Files
- task4.py — Main code
- plot1_sentiment_distribution.png
- plot2_sentiment_by_topic.png
- plot3_sentiment_pie.png
- plot4_wordcloud_positive.png
- plot5_wordcloud_negative.png
- plot6_pos_neg_topics.png

## 🏢 Internship
Prodigy InfoTech Data Science Internship
