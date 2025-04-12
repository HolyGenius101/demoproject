import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import datetime
import requests
from bs4 import BeautifulSoup

# Load sentiment model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['negative', 'neutral', 'positive']

# Analyze sentiments in batches
def analyze_sentiment_batch(texts):
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded)
    scores = F.softmax(output.logits, dim=1).numpy()
    sentiments = [labels[s.argmax()] + f" ({s[2] - s[0]:.2f})" for s in scores]  # label + score
    sentiment_scores = [float(s[2] - s[0]) for s in scores]  # positive - negative
    return sentiments, sentiment_scores

# Fetch headlines using Bing News
headers = {"User-Agent": "Mozilla/5.0"}
def fetch_bing_headlines(topic, date, seen_global, limit):
    formatted_date = date.strftime('%Y-%m-%d')
    url = f"https://www.bing.com/news/search?q={topic}+after:{formatted_date}+before:{formatted_date}&FORM=HDRSC6"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    articles = soup.find_all('a', attrs={'class': 'title'})
    headlines_links = []
    for a in articles:
        text = a.get_text().strip()
        link = a['href']
        if text and text not in seen_global:
            seen_global.add(text)
            headlines_links.append((text, link))
        if len(headlines_links) >= limit:
            break
    return headlines_links

# Streamlit UI
st.set_page_config(page_title="Crossroads Sentiment Tracker", layout="centered")
st.title("🛤️ Crossroads: Sentiment Trends in News Over Time")
st.markdown("Analyze how public sentiment evolves in news headlines and discover emotional crossroads around major topics.")

# User inputs
topic = st.text_input("Enter a topic (e.g. climate change, AI):", "climate change")
start = st.date_input("Start Date", datetime.date(2024, 1, 1))
end = st.date_input("End Date", datetime.date(2024, 1, 5))

if st.button("Analyze"):
    with st.spinner("Scraping and analyzing..."):
        total_days = (end - start).days + 1
        total_cap = 200
        articles_per_day = max(1, total_cap // total_days)

        current = start
        all_data = []
        all_headlines = []
        seen_global = set()
        total_collected = 0

        while current <= end and total_collected < total_cap:
            daily_cap = min(articles_per_day, total_cap - total_collected)
            headlines_links = fetch_bing_headlines(topic, current, seen_global, daily_cap)
            if headlines_links:
                headlines = [t[0] for t in headlines_links]
                links = [t[1] for t in headlines_links]
                sentiments, scores = analyze_sentiment_batch(headlines)
                avg_score = sum(scores) / len(scores)
                all_data.append({"date": current, "avg_score": avg_score})
                for h, s, l in zip(headlines, sentiments, links):
                    hyperlink = f"[{h}]({l})"
                    all_headlines.append({"date": current, "headline": hyperlink, "sentiment": s})
                total_collected += len(headlines_links)
            current += datetime.timedelta(days=1)

        if not all_data:
            st.warning("No headlines found.")
        else:
            df = pd.DataFrame(all_data)
            st.line_chart(df.set_index("date"))
            avg = df['avg_score'].mean()
            mood = "positive" if avg > 0 else "negative" if avg < 0 else "neutral"
            st.markdown(f"**Overall Sentiment:** {mood.title()} (avg score = {avg:.3f})")
            st.markdown(f"**Total Articles Used:** {total_collected}")

            # Display headlines and their sentiment
            st.markdown("### Headlines and Sentiments")
            dfh = pd.DataFrame(all_headlines)
            st.write(dfh.to_markdown(index=False), unsafe_allow_html=True)
