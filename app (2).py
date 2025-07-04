
st.title("Sentiment Analysis on Amazon Product Reviews")

review = st.text_area("Enter a customer review:")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a valid review text.")
    else:
        # Preprocess and predict
        review_vector = vectorizer.transform([review])
        pred_lr = logistic_model.predict(review_vector)[0]
        pred_nb = nb_model.predict(review_vector)[0]

        sentiment_lr = label_encoder.inverse_transform([pred_lr])[0]
        sentiment_nb = label_encoder.inverse_transform([pred_nb])[0]

        st.subheader("Sentiment Predictions:")
        st.markdown(f"🔹 Logistic Regression: **{sentiment_lr.upper()}**")
        st.markdown(f"🔹 Naive Bayes: **{sentiment_nb.upper()}**")
