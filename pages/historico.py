import streamlit as st
import urllib.parse
from src import soporte_bbdd_mongo as sbm
import pandas as pd
from src import soporte_streamlit as ss

# Obtener el nombre del streamer desde la URL
query_params = st.query_params
streamer = query_params.get("streamer", "Desconocido")

ruta_imagen = "imagenes/logo.png"
st.set_page_config(page_title=f"Historial de {streamer}", page_icon=ruta_imagen, layout="wide", initial_sidebar_state="collapsed")

hide_fullscreen_icon = """
        <style>
        button[aria-label="Fullscreen"] {
            display: none;
        }
        header {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        button[kind="headerNoPadding"] {
            display: none;
        }
        </style>
        """
st.markdown(hide_fullscreen_icon, unsafe_allow_html=True)

# Mostrar el nombre del streamer
st.title(f" 📊 Historial de {streamer} en Twitch")

df = pd.DataFrame(sbm.obtener_datos_historicos_streamer(streamer)).drop(columns=["_id", "id_streamer", "nombre"])

if not df.empty:
    ss.mostrar_datos_historicos(df)
else:
    st.warning("No hay historial disponible para este streamer.")

if st.button("⬅ Volver a la página principal"):
    st.switch_page("streamlit_app.py")