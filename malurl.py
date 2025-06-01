import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import os
import gdown
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def download_file_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
    else:
        print(f"{destination} existe déjà. Téléchargement ignoré.")

# === Chargements ===
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("malurl_modele.keras")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_data
def load_data():
    df = pd.read_csv("malicious_phish.csv")
    return df

@st.cache_data
def prepare_label_encoder(df):
    encoder = LabelEncoder()
    encoder.fit(df['type'])
    return encoder

# === Nettoyage URL ===
def clean_url(url):
    url = url.lower()
    return re.sub(r'[^a-z0-9]', ' ', url)

# === Prédiction ===
def predict_url(url, model, tokenizer, encoder):
    clean = clean_url(url)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    label_index = pred.argmax(axis=1)[0]
    return encoder.inverse_transform([label_index])[0]

# === Interface Streamlit ===
st.set_page_config(page_title="Malicious URL Detector", layout="wide")
# === Menu de navigation dans la sidebar ===
menu = st.sidebar.radio(
    "📌 Menu de navigation",
    [
        "Accueil",
        "Prédiction d'une URL ",
        "Prédictions sur 15 URLs aléatoires"
    ]
)

        
# === Téléchargement des fichiers depuis Google Drive ===

download_file_from_gdrive("1-AIPtEak9q7fDJJU3JN_-qFeeHH3x5Tx", "malurl_modele.keras")
download_file_from_gdrive("1R1Nr1RLA81QC22t7MXrmkMQv0if-v5Oh", "malicious_phish.csv")
download_file_from_gdrive("10SqRzMMnbSzs9XWllbNYKBKWQjF2i2qw", "tokenizer.pkl")


# Chargement des composants
model, tokenizer = load_model_and_tokenizer()
df = load_data()
label_encoder = prepare_label_encoder(df)

if menu == "Accueil":
    st.title("🔍 Détection d'URLs malveillantes avec LSTM + GloVe")
    st.markdown("""
    Cette application utilise un modèle LSTM pour prédire la catégorie d'une URL. 
    Elle permet de visualiser la distribution des classes, 
    tester une URL aléatoire et examiner les prédictions sur un échantillon de 15 URLs.
    """)


    # Affichage conditionnel du DataFrame
    if st.checkbox("Afficher le DataFrame brut"):
        st.dataframe(df.head(100))

    # Répartition des classes
    st.subheader("📊 Répartition des classes")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="type", order=df["type"].value_counts().index, palette="viridis", ax=ax)
    ax.set_title("Nombre d'exemples par classe")
    st.pyplot(fig)
    
    
elif menu == "Prédiction d'une URL ":
    # === Prédiction sur un URL aléatoire ===
    st.subheader("🔎 Prédiction sur une URL aléatoire")
    if st.button("🎲 Tirer une URL aléatoire"):
        sample = df.sample(1).iloc[0]
        url = sample['url']
        true_label = sample['type']
        pred_label = predict_url(url, model, tokenizer, label_encoder)

        col1, col2 = st.columns(2)
        col1.markdown(f"**URL brut :** {url}")
        col1.markdown(f"**Classe réelle :** {true_label}")
        col1.markdown(f"**Classe prédite :** {pred_label}")

        if pred_label == true_label:
            col2.success("✅ Prédiction correcte")
        else:
            col2.warning("⚠️ Prédiction incorrecte")
            
            
    # === Prédiction manuelle d'une URL ===
    st.subheader("🧪 Tester une URL personnalisée")

    user_url = st.text_input("Entrez une URL à tester :", placeholder="https://example.com/malicious")

    if st.button("🔍 Prédire cette URL"):
        if user_url.strip() == "":
            st.warning("❗ Veuillez entrer une URL valide.")
        else:
            pred_label = predict_url(user_url, model, tokenizer, label_encoder)
            st.markdown(f"**Classe prédite :** `{pred_label}`")


# === Prédictions sur 15 URLs aléatoires ===
elif menu == "Prédictions sur 15 URLs aléatoires":
    st.subheader("📋 Prédiction sur 15 URLs selectionnées aléatoirement dans la base de test")
    if st.button("🎲 Tirer 15 URLs aléatoires"):
        sample_df = df.sample(15).reset_index(drop=True)
        sample_df['predicted_label'] = sample_df['url'].apply(lambda x: predict_url(x, model, tokenizer, label_encoder))
        sample_df['correct'] = sample_df['type'] == sample_df['predicted_label']

        def highlight_incorrect(row):
            color = 'background-color: salmon'
            return [color if not row['correct'] and col == 'correct' else '' for col in row.index]

        st.dataframe(sample_df[['url', 'type', 'predicted_label', 'correct']].style.apply(highlight_incorrect, axis=1))


