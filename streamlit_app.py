import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src import soporte_bbdd_mongo as sbm
import warnings
warnings.filterwarnings("ignore")

# Interfaz Streamlit
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("imagenes/logo.png", use_container_width=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Streamers Finder</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <h3 style="text-align: center;"> 🎮 Descubre los Mejores Streamers para Ti 🎮</h3>

    #### 🔍 ¿De qué trata esta app?
    
    Bienvenido/a a **tu guía personalizada de streamers**. 📺🎮

    Aquí podrás **descubrir y explorar** streamers basados en las categorías de juegos que te gustan.

    🚀 Encuentra contenido nuevo, compara streamers y accede a métricas detalladas para tomar la mejor decisión.  

    🔥 **¿Listo para descubrir tu próximo streamer favorito?** ¡Empecemos! 🎧✨
    """,
    unsafe_allow_html=True
)
st.write("""
🤖 Nuestro sistema de recomendación tiene dos enfoques:

**📢 Modo Básico:** Te sugiere streamers populares. ¿Por qué? Porque es más fácil que disfrutes contenido de alguien con una comunidad establecida y miles de seguidores.

**🌱 Modo Avanzado:** Aquí la cosa cambia. No solo miramos los grandes nombres, sino que también te recomendamos streamers más pequeños que pueden encajar con lo que buscas. Así apoyamos a creadores emergentes y te ayudamos a descubrir nuevas joyas en Twitch. 🎥✨

Elige el modo que más te guste y descubre contenido nuevo a tu medida. 🚀
""")

# Conectar a MongoDB
client, db = sbm.conectar_mongo()

# Cargar datos
@st.cache_data
def load_data():
    historico = pd.DataFrame(list(db.historico_streamers_total.find())).drop(columns=["_id"])
    categorias = pd.DataFrame(list(db.categorias_streameadas.find())).drop(columns=["_id"])
    streamers = pd.DataFrame(list(db.streamers.find())).drop(columns=["_id"])  # Cargar ranking
    return historico, categorias, streamers

historico, categorias, streamers = load_data()

client.close()

modelos_knn_basico, num_cols_basico, scaler_basico, kmeans_basico, le_basico = joblib.load("modelos/modelo_basico.pkl")
modelos_knn_avanzado, num_cols_avanzado, kmeans_avanzado, scaler_avanzado, le_avanzado = joblib.load("modelos/modelo_lastversion.pkl")

# Unir datasets
# historico = historico.drop(columns=['_id'])
# categorias = categorias.drop(columns=['_id'])
# streamers = streamers.drop(columns=['_id'])
df = historico[["id_streamer", "avgviewers", "peakviewers", "hoursstreamed"]].merge(categorias, on='id_streamer', how='left')
df = df.merge(streamers[['id_streamer', 'rank', 'all_time_peak_viewers', 'total_followers', 'total_views']], on='id_streamer', how='left')  # Agregar ranking
df = df[df['categoria'] != "18+"]
df = df[df['categoria'] != "171"]
df = df[df['categoria'] != "2XKO"]
df = df[df['categoria'] != "1v1.LOL"]
df_original = df.copy()

# df["rank"] = df["rank"].fillna(df["rank"].median())
df["rank"] = df["rank"].max() + df["rank"].min() - df["rank"]
df["rank"] = df["rank"].astype(np.int32)


def recomendar_streamers_por_categoria_avanzado(categoria: str, df: pd.DataFrame, n=5):
    df_categoria = df[df['categoria'] == categoria]

    if df_categoria.empty:
        print(f"⚠️ No hay streamers en la categoría {categoria}")
        return pd.DataFrame()

    # 🔹 Ordenar por ranking ya invertido (ascending=False es correcto en este caso)
    df_categoria_ordenado = df_categoria.sort_values(by=['rank'], ascending=False)  

    # ✅ Devolver los `n` mejores sin aplicar KNN
    return df_categoria_ordenado[['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank']].head(n)


def recomendar_streamers_por_categoria(categoria, df, n=5):
    if categoria not in modelos_knn_basico:
        return pd.DataFrame()
    
    df_categoria = df[df['categoria'] == categoria]
    if df_categoria.empty:
        return pd.DataFrame()
    
    X_categoria = df_categoria[num_cols_basico]
    nn = modelos_knn_basico[categoria]
    
    centroide = X_categoria.mean().values.reshape(1, -1)
    distancias, indices = nn.kneighbors(centroide)
    
    columnas_validas = ['id_streamer', 'nombre', 'categoria', 'rank']
    
    recomendaciones = df_categoria.iloc[indices[0]][columnas_validas]
    
    recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer')
    recomendaciones = recomendaciones.sort_values(by='rank', ascending=False)  # Ordenar por ranking
    
    return recomendaciones

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

    st.markdown(tabla_html, unsafe_allow_html=True)


# # Interfaz Streamlit
# st.title("Recomendador de Streamers por Categoría")
# st.write("Este recomendador utiliza un modelo de K-Nearest Neighbors para encontrar streamers similares a los de una categoría seleccionada.")

# Lista de categorías únicas
categorias_unicas = df['categoria'].drop_duplicates().sort_values().tolist()
categoria_seleccionada = st.selectbox("Seleccione una categoría:", categorias_unicas)

radio = st.radio("Elige modelo:", ('Basico', 'Avanzado'))

if st.button("Recomendar"):
    if radio == 'Avanzado':
        # num_cols_basico.remove('categoria_codificada')
        # num_cols_basico.remove('cluster')
        # Cargar modelos KNN entrenados
        
        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols_basico:
            col = str(col)
            df[col] = df[col].astype(np.float32)
        
        # Ponderar columnas clave antes de normalizar
        df['rank'] = df['rank'] * 5  # Dar más peso al ranking
        df['total_followers'] = df['total_followers'] * 3  # Más peso a seguidores
        df['total_views'] = df['total_views'] * 3  # Más peso a vistas totales

        df['categoria_codificada'] = le_basico.fit_transform(df['categoria'])

        # num_cols_basico.append('categoria_codificada')

        df["cluster"] = kmeans_basico.predict(df[num_cols_basico])

        # num_cols_basico.append('cluster')

        df[num_cols_basico] = scaler_basico.fit_transform(df[num_cols_basico])
        resultados = recomendar_streamers_por_categoria(categoria_seleccionada, df, num_cols_basico)
    else:
        num_cols_avanzado.remove('categoria_codificada')
        num_cols_avanzado.remove('cluster')
        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols_avanzado:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['categoria_codificada'] = le_avanzado.fit_transform(df['categoria'])

        num_cols_avanzado.append('categoria_codificada')

        # Ponderar columnas clave antes de normalizar
        df['rank'] *= 5  # Dar más peso al ranking
        df['total_followers'] *= 3  # Más peso a seguidores
        df['total_views'] *= 3  # Más peso a vistas totales

        df[num_cols_avanzado] = scaler_avanzado.fit_transform(df[num_cols_avanzado])

        df["cluster"] = kmeans_avanzado.predict(df[num_cols_avanzado])

        num_cols_avanzado.append('cluster')

        resultados = recomendar_streamers_por_categoria_avanzado(categoria_seleccionada, df)
    if not resultados.empty:
        st.write(f"### Streamers Recomendados para la Categoría: {categoria_seleccionada}")
    #     st.dataframe(
    # resultados.drop(columns=["id_streamer"])
    # .rename(columns={'nombre': 'Nombre', 'categoria': 'Categoría', 'total_followers': 'Seguidores', 'total_views': 'Vistas Totales', 'rank': 'Ranking'})
    # .reset_index(drop=True)
    # .head(5))
        df_nuevo = pd.DataFrame()
        for indice, fila in resultados.iterrows():
            df_nuevo = pd.concat([df_nuevo, df_original[df_original.index == indice]], axis=0)
        df_nuevo = df_nuevo.reindex(columns=["nombre"] + df_nuevo.columns.drop(["nombre", "id_streamer", "avgviewers", "peakviewers", "categoria", "rank"]).to_list() + ["rank"])
        df_nuevo = df_nuevo.rename(columns={
            "nombre": "Nombre",
            "hoursstreamed": "Horas Streameadas",
            "all_time_peak_viewers": "Pico Máx. Espectadores",
            "total_followers": "Total Seguidores",
            "total_views": "Total Vistas",
            "rank": "Ranking"
        })
        # st.dataframe(df_nuevo)
        mostrar_datos(df_nuevo)

    else:
        st.write("No se encontraron recomendaciones para la categoría seleccionada.")
