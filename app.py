import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Cargar el modelo y el escalador
scaler = joblib.load("escalador.bin")
model = joblib.load("modelo_knn.bin")

# Definir las URLs de las imágenes
imagen_si_url = "https://i.pinimg.com/474x/1a/87/bd/1a87bd2d4bdb90f94d068fdf69179446.jpg"
imagen_no_url = "https://m.media-amazon.com/images/I/41QQ1gs13dL.jpg"

def cargar_imagen(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Configuración de la aplicación
st.title("Asistente para cardiólogos")

# Crear pestañas
tabs = st.tabs(["Instrucciones", "Captura de Datos", "Predicción"])

with tabs[0]:
    st.header("Instrucciones")
    st.write("Esta aplicación permite predecir si una persona tiene problemas cardíacos en función de la edad y el nivel de colesterol.")
    st.write("Simplemente ingrese los datos en la pestaña de Captura de Datos y luego realice la predicción en la pestaña de Predicción.")
    st.write("- **Edad**: debe estar entre 18 y 80 años.")
    st.write("- **Colesterol**: debe estar entre 50 y 600 mg/dL.")

with tabs[1]:
    st.header("Captura de Datos")
    edad = st.number_input("Ingrese su edad", min_value=18, max_value=80, step=1)
    colesterol = st.number_input("Ingrese su nivel de colesterol", min_value=50, max_value=600, step=1)
    st.session_state["edad"] = edad
    st.session_state["colesterol"] = colesterol

with tabs[2]:
    st.header("Predicción de problemas cardíacos")
    
    if st.button("Predecir"):
        # Obtener datos de sesión
        edad = st.session_state.get("edad", 18)
        colesterol = st.session_state.get("colesterol", 50)
        
        # Crear un DataFrame con los nombres de las columnas correctas
        datos_df = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])
        
        # Normalizar los datos
        datos_norm = scaler.transform(datos_df)
        
        # Realizar predicción
        prediccion = model.predict(datos_norm)[0]
        
        # Mostrar resultado
        if prediccion == 1:
            st.error("Se detecta un posible problema cardíaco.")
            st.image(cargar_imagen(imagen_si_url), caption="¡Atención! Consulte a su médico.")
        else:
            st.success("No se detectan problemas cardíacos.")
            st.image(cargar_imagen(imagen_no_url), caption="¡Buena salud! Siga cuidándose.")
