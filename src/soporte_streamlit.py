from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi
import pandas as pd
import numpy as np
import streamlit as st
from src import soporte_bbdd_mongo as sbm



# Cargar datos
@st.cache_data(show_spinner="🔄 Procesando datos, un momento...", ttl=21600)
def load_data(_db):
    historico = pd.DataFrame(list(_db.historico_streamers_total.find())).drop(columns=["_id"])
    categorias = pd.DataFrame(list(_db.categorias_streameadas.find())).drop(columns=["_id"])
    streamers = pd.DataFrame(list(_db.streamers.find())).drop(columns=["_id"])  # Cargar ranking
    return historico, categorias, streamers

# Función para mostrar los datos con formato en Streamlit
def mostrar_datos(df):
    st.markdown("""
    <style>
        .tabla-custom {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
            text-align: center;
        }
        .tabla-custom th {
            background-color: #4b9fff;
            color: white;
            padding: 12px;
            border: 1px solid #ddd;
        }
        .tabla-custom td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        .tabla-custom tr:nth-child(even) {
            background-color: black;
        }
        .tabla-custom tr:hover {
            background-color: gray;
        }
    </style>
    """, unsafe_allow_html=True)

    tabla_html = "<table class='tabla-custom'>"
    tabla_html += "<tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr>"

    for _, row in df.iterrows():
        fila = "<tr>"
        for col in df.columns:
            valor = row[col]
            if isinstance(valor, (int, float)):  # Aplicar formato solo si es numérico
                fila += f"<td>{valor:,.0f}</td>"  # Separador de miles sin decimales
            else:
                fila += f"<td>{valor}</td>"
        fila += "</tr>"
        tabla_html += fila

    tabla_html += "</table>"

    return st.markdown(tabla_html, unsafe_allow_html=True)

if __name__ == "__main__":
    load_data()