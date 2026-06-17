import streamlit as st
import pickle

st.set_page_config(page_title="Sentimen Review App", page_icon="🎯")

st.title("🎯 Prediksi Sentimen Review Aplikasi")
st.write("Masukkan teks review aplikasi di bawah, model akan memprediksi sentimennya.")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.success("✅ Model siap digunakan!")

st.header("🔍 Prediksi Sentimen")
teks = st.text_area("Tulis review aplikasi di sini:", height=150)

if st.button("Prediksi Sentimen"):
    if teks.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        X_input = vectorizer.transform([teks])
        hasil = model.predict(X_input)[0]

        if hasil == "Positive":
            st.success(f"✅ Sentimen: **{hasil}**")
        elif hasil == "Negative":
            st.error(f"❌ Sentimen: **{hasil}**")
        else:
            st.info(f"➖ Sentimen: **{hasil}**")

st.divider()
st.caption("Dataset: Google Play Store Reviews (Kaggle) | Model: Naive Bayes + TF-IDF")
