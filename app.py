import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load your custom DeBERTa model from Hugging Face Hub
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis", model="Siddharth-Adhikari-07/finetuned-deberta-sentiment")

classifier = load_pipeline()

# App UI
st.title("🧠 Sentiment Wars")
st.markdown("Enter a sentence and visualize model confidence across sentiment classes.")

user_input = st.text_area("✍️ Enter a sentence to analyze sentiment:")

if st.button("🔍 Analyze"):
    if user_input.strip():
        results = classifier([user_input])

        label = results[0]["label"]
        score = round(results[0]["score"] * 100, 2)

        # Show prediction
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score}%")

        # 📊 Show pie chart
        st.subheader("🍰 Sentiment Distribution (Model Confidence)")

        labels = [result["label"] for result in results]
        scores = [result["score"] * 100 for result in results]

        fig, ax = plt.subplots()
        ax.pie(scores, labels=labels, autopct="%1.1f%%", startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
        ax.axis("equal")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter some text to analyze.")
