import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

st.set_page_config(page_title="Sentimen Review App", page_icon="🎯")

st.title("🎯 Prediksi Sentimen Review Aplikasi")
st.write("Upload dataset, training model, lalu prediksi sentimen review secara real-time.")

# ── Section 1: Upload Dataset ──
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV (googleplaystore_user_reviews.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['Translated_Review', 'Sentiment'])
    
    st.success(f"Dataset berhasil dimuat! Total data: {len(df)} baris")
    st.dataframe(df[['Translated_Review', 'Sentiment']].head(5))

    # ── Section 2: Training Model ──
    st.header("2️⃣ Training Model Naive Bayes")

    X = df['Translated_Review']
    y = df['Sentiment']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']

    st.success(f"✅ Model berhasil ditraining! Akurasi: {accuracy:.2%}")

    # ── Section 3: Prediksi ──
    st.header("3️⃣ Prediksi Sentimen")
    teks = st.text_area("Masukkan teks review aplikasi:", height=150)

    if st.button("🔍 Prediksi Sentimen"):
        if teks.strip() == "":
            st.warning("Teks review tidak boleh kosong!")
        else:
            X_input = vectorizer.transform([teks])
            hasil = model.predict(X_input)[0]

            if hasil == "Positive":
                st.success(f"✅ Sentimen: **{hasil}**")
            elif hasil == "Negative":
                st.error(f"❌ Sentimen: **{hasil}**")
            else:
                st.info(f"➖ Sentimen: **{hasil}**")
