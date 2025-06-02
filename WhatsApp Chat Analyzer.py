import re
import pandas as pd
import streamlit as st
from transformers import pipeline
import plotly.express as px
from datetime import datetime


# 1. WhatsApp Parser (Handles all date formats)
def parse_whatsapp_chat(file_content):
    pattern = r'(\d+/\d+/\d+,\s\d+:\d+\s?[APMapm]*)\s-\s([^:]+):\s(.+)'
    messages = re.findall(pattern, file_content)
    df = pd.DataFrame(messages, columns=["timestamp", "sender", "text"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    return df


# 2. Emotion Analysis (Optimized model)
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis")


# 3. Streamlit App
st.set_page_config(layout="wide")
st.title("📊 WhatsApp Chat Analyzer")

uploaded_file = st.file_uploader("WhatsApp Chat with Group Study.txt", type="txt")

if uploaded_file:
    # Parse chat
    content = uploaded_file.getvalue().decode("utf-8")
    df = parse_whatsapp_chat(content)
    
    if not df.empty:
        st.success(f"✅ Analyzed {len(df)} messages")
        
        # Analyze first 300 messages (for speed)
        sample_df = df.head(300).copy()
# Get emotions
        emotion_model = load_model()
        sample_df["emotion"] = sample_df["text"].apply(
            lambda x: emotion_model(x[:512])[0]["label"] if x.strip() else "neutral"
        )
        
        # 1. Emotion Pie Chart
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("😊 Emotional Breakdown")
            fig1 = px.pie(sample_df, names="emotion")
            st.plotly_chart(fig1, use_container_width=True)
# 2. Activity Timeline
        with col2:
            st.subheader("⏰ Message Activity")
            timeline = sample_df.set_index("timestamp").resample("H").size()
            fig2 = px.line(timeline, title="Messages per Hour")
            st.plotly_chart(fig2, use_container_width=True)
        
        # 3. Sender Stats
        st.subheader("👥 Sender Analysis")
        tab1, tab2 = st.tabs(["Emotions", "Activity"])
        
        with tab1:
            fig3 = px.bar(
sample_df.groupby(["sender", "emotion"]).size().reset_index(name="count"),
                x="sender", y="count", color="emotion",
                title="Who Feels What?"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            fig4 = px.bar(
                sample_df["sender"].value_counts().reset_index(),
                x="index", y="sender",
                title="Most Active Members"
            )
            st.plotly_chart(fig4, use_container_width=True)
# Download button
        st.download_button(
            "📥 Download Full Analysis",
            sample_df.to_csv(index=False),
            file_name="whatsapp_analysis.csv"
        )
    else:
        st.error("No messages found! Export chat correctly.")
else:
    st.info("ℹ How to export: WhatsApp → Chat → ⋮ → More → Export chat (Without media)")