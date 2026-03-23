import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="Sentiment Analyzer PRO",
    page_icon="🎬",
    layout="wide"
)

uploaded_file = st.file_uploader("📂 Upload a file (.txt or .csv)", type=["txt", "csv"])

# Load model + vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
nb_model = pickle.load(open("model/nb_model.pkl","rb"))

st.markdown("""
# 🎬 Sentiment Analyzer PRO  
Analyze movie reviews with AI-powered sentiment detection
""")

st.markdown("### Enter a movie review and get instant sentiment analysis 🎯")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

review = st.text_area(
    "Type your movie review here",
    key="review_text"
)

if review and len(review.split()) < 3:
    st.warning("⚠️ Please enter a slightly longer review for better accuracy")

if st.button("Clear"):
    if "review_text" in st.session_state:
        st.session_state.pop("review_text")
    if "history" in st.session_state:
        st.session_state.history = []
    st.rerun()

reviews_list = []

# If file uploaded
if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        reviews_list = content.split("\n")

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        reviews_list = df.iloc[:, 0].dropna().tolist()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Run automatically (real-time)
if (review and review.strip()) or reviews_list:

    if reviews_list:

        st.markdown("---")
        st.subheader("📊 Batch Analysis Results")

        results = []

        for r in reviews_list:
            if r.strip() == "":
                continue

            vec = vectorizer.transform([r])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec).max() * 100

            results.append({
                "Review": r,
                "Prediction": "Positive 😊" if pred == 1 else "Negative 😡",
                "Confidence (%)": round(prob, 2)
            })

        df_results = pd.DataFrame(results)

        st.dataframe(df_results)

        st.markdown("### 🔍 Select a review for detailed analysis")

        selected_review = st.selectbox(
            "Choose a review",
            df_results["Review"]
        )

        if selected_review:

            st.markdown("---")
            st.subheader("🧠 Detailed Analysis")

            vec = vectorizer.transform([selected_review])

            prediction = model.predict(vec)
            probability = model.predict_proba(vec)

            confidence = probability.max() * 100

            st.write("Prediction:",
                    "Positive 😊" if prediction[0]==1 else "Negative 😡")

            st.metric("Confidence", f"{round(confidence,2)}%")

            # Chart
            positive_score = probability[0][1]
            negative_score = probability[0][0]

            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Score": [positive_score, negative_score]
            })

            st.bar_chart(chart_data.set_index("Sentiment"))

            # Highlighting
            feature_names = vectorizer.get_feature_names_out()
            weights = model.coef_[0]

            words = selected_review.split()
            highlighted_text = ""

            word_index = {word: i for i, word in enumerate(feature_names)}

            for word in words:
                clean_word = word.lower().strip(".,!?")
                index = word_index.get(clean_word)

                if index is not None:
                    weight = weights[index]

                    if weight > 0:
                        highlighted_text += f" **:green[{word}]**"
                    else:
                        highlighted_text += f" **:red[{word}]**"
                else:
                    highlighted_text += f" {word}"

            st.markdown(highlighted_text)

        # Summary
        positive_count = sum(1 for r in results if "Positive" in r["Prediction"])
        negative_count = sum(1 for r in results if "Negative" in r["Prediction"])

        st.markdown("### 📈 Summary")

        summary_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive_count, negative_count]
        })      

        st.bar_chart(summary_df.set_index("Sentiment"))

        st.write(f"Total Reviews: {len(results)}")
        st.write(f"Positive reviews: {positive_count}")
        st.write(f"Negative reviews: {negative_count}")

    elif review and review.strip():

        vector = vectorizer.transform([review])

        # Logistic prediction
        prediction = model.predict(vector)
        probability = model.predict_proba(vector)

        confidence = probability.max() * 100

        # Confidence interpretation
        if confidence > 80:
            st.success("High confidence prediction ✅")
        elif confidence > 60:
            st.warning("Moderate confidence ⚠️")
        else:
            st.error("Low confidence - uncertain prediction ❗")

        # Naive Bayes prediction (simulated)
        nb_pred = nb_model.predict(vector)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Prediction")

            st.write("Logistic:",
                    "Positive 😊" if prediction[0]==1 else "Negative 😡")

            st.write("Naive Bayes:",
                    "Positive 😊" if nb_pred[0]==1 else "Negative 😡")

        with col2:
            st.subheader("📊 Confidence")

            st.metric("Confidence Score", f"{round(confidence,2)}%")

            if prediction[0] == 1:
                st.success("Positive")
            else:
                st.error("Negative")

        if prediction[0] == nb_pred[0]:
            st.success("✔ Both models agree on the prediction")
        else:
            st.warning("⚠️ Models disagree - prediction may be uncertain")

        
        st.markdown("---")
        st.subheader("📈 Sentiment Breakdown")

        # Create labeled data
        positive_score = probability[0][1]
        negative_score = probability[0][0]

        # Show percentage text (clean + clear)
        st.write(f"🟢 Positive: {round(positive_score*100,2)}%")
        st.write(f"🔴 Negative: {round(negative_score*100,2)}%")

        # Create labeled DataFrame for bar chart
        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Score": [positive_score, negative_score]
        })

        st.bar_chart(chart_data.set_index("Sentiment"))

        


        # Highlight words
        feature_names = vectorizer.get_feature_names_out()
        weights = model.coef_[0]

        st.subheader("🧠 Sentiment Word Highlighting")

        words = review.split()
        highlighted_text = ""
        word_index = {word: i for i, word in enumerate(feature_names)}

        for word in words:
            clean_word = word.lower().strip(".,!?")
            index = word_index.get(clean_word)
            if index is not None:
                weight = weights[index]

                if weight > 0:
                    highlighted_text += f" **:green[{word}]**"
                else:
                    highlighted_text += f" **:red[{word}]**"
            else:
                highlighted_text += f" {word}"

        st.markdown(highlighted_text)
        st.info("🟢 Green words → Positive influence | 🔴 Red words → Negative influence")

        if review not in st.session_state.history:
            st.session_state.history.append(review)

# Show history
if st.session_state.history:
    st.subheader("🕘 Review History")
    for r in st.session_state.history[-5:]:
        st.write("-", r)