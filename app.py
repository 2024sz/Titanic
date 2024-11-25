
import streamlit as st
import pandas as pd
import joblib

# Carica il modello
@st.cache_resource
def load_model():
    with open("best_grid_model.pkl", "rb") as file:
        model = joblib.load(file)
    return model

model = load_model()

# Titolo dell'app
st.title("App di Predizione con Machine Learning")
st.write("Carica i dati o inserisci manualmente le informazioni per effettuare una predizione.")

# Input dell'utente
st.sidebar.header("Inserisci i dati per la predizione:")
pclass = st.sidebar.selectbox("Classe del Passeggero (Pclass)", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sesso (0 = Femmina, 1 = Maschio)", [0, 1], index=1)
age = st.sidebar.slider("Età (Age)", min_value=0, max_value=100, value=25, step=1)
sibsp = st.sidebar.slider("Numero di Fratelli/Coniugi a Bordo (SibSp)", min_value=0, max_value=10, value=0, step=1)
parch = st.sidebar.slider("Numero di Genitori/Figli a Bordo (Parch)", min_value=0, max_value=10, value=0, step=1)
fare = st.sidebar.number_input("Tariffa Pagata (Fare)", min_value=0.0, value=10.0, step=0.1)
embarked = st.sidebar.selectbox("Porto d'Imbarco (Embarked: 0 = Cherbourg, 1 = Queenstown, 2 = Southampton)", [0, 1, 2], index=2)

# Predizione
new_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

if st.sidebar.button("Calcola Predizione"):
    prediction = model.predict(new_data)
    st.subheader("Risultati della Predizione")
    if prediction[0] == 1:
        st.success("Il passeggero ha probabilmente sopravvissuto!")
    else:
        st.error("Il passeggero non ha probabilmente sopravvissuto.")

# Suggerimenti
st.sidebar.markdown("**Suggerimento**: Modifica i parametri per vedere come cambia la predizione.")
