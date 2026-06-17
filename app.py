import streamlit as st
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# UI
st.title("🎯 Prediksi Sentimen Review Aplikasi")
st.write("Masukkan teks review aplikasi, dan model akan memprediksi sentimennya.")

teks = st.text_area("Tulis review di sini:", height=150)

if st.button("Prediksi Sentimen"):
    if teks.strip() == "":
        st.warning("Teks review tidak boleh kosong!")
    else:
        X_input = vectorizer.transform([teks])
        hasil = model.predict(X_input)[0]
        
        if hasil == "Positive":
            st.success(f"✅ Sentimen: {hasil}")
        elif hasil == "Negative":
            st.error(f"❌ Sentimen: {hasil}")
        else:
            st.info(f"➖ Sentimen: {hasil}")