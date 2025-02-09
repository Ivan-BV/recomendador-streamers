import streamlit as st
import pandas as pd
import numpy as np
from src.soporte_bbdd_mongo import conectar_mongo
from src import soporte_streamlit as ss
import warnings
warnings.filterwarnings("ignore")
import pickle

ruta_imagen = "imagenes/logo.png"
st.set_page_config(page_title="Streamer Finder", page_icon=ruta_imagen, initial_sidebar_state="collapsed")

# Interfaz Streamlit
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
        a {
            color: white;
            text-decoration: none;
        }
        </style>
        """
st.markdown(hide_fullscreen_icon, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.image(ruta_imagen, use_container_width=True)

col1, col2, col3 = st.columns([2,18,1])
with col2:
    st.markdown("<h3 style='text-align: center;'> 🎮 Descubre los Mejores Streamers para Ti 🎮 </h3>", unsafe_allow_html=True)

st.write(
    """
    ###
    
    📺 Bienvenido/a a **tu recomendador de streamers**.

    🧑‍💻 Aquí podrás **descubrir y explorar** streamers basados en las categorías de juegos que te gustan.

    🚀 Encuentra contenido nuevo y accede a métricas detalladas para tomar la mejor decisión.  

    🔥 **¿Listo para descubrir tu próximo streamer favorito?** ¡Empecemos! 🎧✨

    🤖 Nuestro sistema de recomendación tiene dos enfoques:

    **📢 Modo Básico:** Te sugiere streamers populares. ¿Por qué? Porque es más fácil que disfrutes contenido de alguien con una comunidad establecida y miles de seguidores.

    **🌱 Modo Avanzado:** Aquí la cosa cambia. No solo miramos los grandes nombres, sino que también te recomendamos streamers más pequeños que pueden encajar con lo que buscas. Así apoyamos a creadores emergentes y te ayudamos a descubrir nuevas joyas en Twitch. 🎥✨

    ⚙️ Elige el modo que más te guste y descubre contenido nuevo a tu medida.
    """
)

try:
    client, db = conectar_mongo()

    historico, categorias, streamers = ss.load_data(db)

    client.close()
except:
    st.error("Error al cargar los datos")

df = ss.preprocesar_datos(historico, categorias, streamers)
df_original = df.copy()

categorias_unicas = df['categoria'].drop_duplicates().tolist()
categoria_seleccionada = st.selectbox("Selecciona una categoría:", categorias_unicas)

try:
    client, db = conectar_mongo()

    modelo_basico, modelo_avanzado = ss.load_models(db)

    client.close()
except:
    st.error("Error al cargar los modelos")

modelos_knn_basico, num_cols_basico, scaler_basico, kmeans_basico, le_basico = pickle.loads(modelo_basico)
modelos_knn_avanzado, num_cols_avanzado, kmeans_avanzado, scaler_avanzado, le_avanzado = pickle.loads(modelo_avanzado)

radio = st.radio("Elige el modo:", ('Basico', 'Avanzado'))

if st.button("Recomendar"):

    if radio == 'Avanzado':
        
        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols_basico:
            col = str(col)
            df[col] = df[col].astype(np.float32)
        
        # Ponderar columnas clave antes de normalizar
        df['rank'] = df['rank'] * 5
        df['total_followers'] = df['total_followers'] * 3
        df['total_views'] = df['total_views'] * 3

        df['categoria_codificada'] = le_basico.fit_transform(df['categoria'])

        df["cluster"] = kmeans_basico.predict(df[num_cols_basico])

        df[num_cols_basico] = scaler_basico.fit_transform(df[num_cols_basico])
        resultados = ss.recomendar_streamers_por_categoria(categoria_seleccionada, df, modelos_knn_basico, num_cols_basico)
    else:
        # Eliminamos las columnas antes del preprocesamiento
        num_cols_avanzado.remove('categoria_codificada')
        num_cols_avanzado.remove('cluster')

        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols_avanzado:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['categoria_codificada'] = le_avanzado.fit_transform(df['categoria'])

        num_cols_avanzado.append('categoria_codificada')

        # Ponderar columnas clave antes de normalizar
        df['rank'] *= 5
        df['total_followers'] *= 3
        df['total_views'] *= 3

        df[num_cols_avanzado] = scaler_avanzado.fit_transform(df[num_cols_avanzado])

        df["cluster"] = kmeans_avanzado.predict(df[num_cols_avanzado])

        num_cols_avanzado.append('cluster')

        resultados = ss.recomendar_streamers_por_categoria_avanzado(categoria_seleccionada, df, modelos_knn_avanzado, num_cols_avanzado)
    if not resultados.empty:
        st.write(f"### Streamers Recomendados para la Categoría: {categoria_seleccionada}")
        st.write(f"Haz clic en el nombre del streamer que quieras consultar su historico")
        df_nuevo = pd.DataFrame()
        resultados = resultados.head(5)
        for indice, fila in resultados.iterrows():
            df_nuevo = pd.concat([df_nuevo, df_original[df_original.index == indice]], axis=0)
        ss.mostrar_datos(df_nuevo)
    else:
        st.write(f"##### No se encontraron recomendaciones para la categoría seleccionada.")
