import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import joblib
from src import soporte_bbdd_mongo as sbm

# Conectar a MongoDB
client, db = sbm.conectar_mongo()

# Cargar datos
historico = pd.DataFrame(list(db.historico_streamers_total.find()))
categorias = pd.DataFrame(list(db.categorias_streameadas.find()))
streamers = pd.DataFrame(list(db.streamers.find()))  # Cargar ranking

client.close()

# Unir datasets
historico = historico.drop(columns=['_id'])
categorias = categorias.drop(columns=['_id'])
streamers = streamers.drop(columns=['_id'])
df = historico[["id_streamer", "avgviewers", "peakviewers", "hoursstreamed"]].merge(categorias, on='id_streamer', how='left')
df = df.merge(streamers[['id_streamer', 'rank', 'all_time_peak_viewers', 'total_followers', 'total_views']], on='id_streamer', how='left')  # Agregar ranking

# df["rank"] = df["rank"].fillna(df["rank"].median())
df["rank"] = df["rank"].max() - df["rank"]

def recomendar_streamers_por_categoria(categoria: str, df: pd.DataFrame, radio, num_cols: list, n=5):
    if categoria not in modelos_knn:
        return pd.DataFrame()
    
    df_categoria = df[df['categoria'] == categoria]
    if df_categoria.empty:
        return pd.DataFrame()
    
    # if radio == 'Avanzado':
    #     # Identificar el clúster del streamer con mejor ranking en la categoría
    #     cluster_top = df_categoria.sort_values(by=['rank'], ascending=False)['cluster'].iloc[0]
        
    #     # siguiente_cluster = df_categoria.sort_values(by=['rank'], ascending=False)['cluster'].iloc[1]

    #     # Filtrar por el clúster top
    #     df_categoria_cluster = df_categoria[df_categoria['cluster'] == cluster_top]

    #     # df_categoria_siguiente_cluster = df_categoria[df_categoria['cluster'] == siguiente_cluster]

    #     # 🚀 Filtrar solo los streamers con rank alto dentro del clúster
    #     df_categoria_cluster_top = df_categoria_cluster.sort_values(by=['rank'], ascending=False).head(10)

    #     # Si hay pocos streamers en el clúster, usar toda la categoría
    #     if df_categoria_cluster_top.shape[0] < 5:
    #         # if not df_categoria_siguiente_cluster.empty:
    #         #     print("Usando siguiente cluster")
    #         #     df_categoria_siguiente_cluster = df_categoria_siguiente_cluster.sort_values(by=['rank'], ascending=False).head(10)
    #         #     # df_categoria_cluster_top = pd.concat([df_categoria_cluster_top, df_categoria_siguiente_cluster]).drop_duplicates(subset='id_streamer', keep='first')
    #         # else:
    #         #     df_categoria_cluster_top = df_categoria_cluster
    #         df_categoria_cluster_top = df_categoria_cluster

    #     # Aplicar KNN dentro del clúster
    #     X_categoria = df_categoria_cluster_top[num_cols].reset_index(drop=True)  # Resetear índices para evitar problemas
    #     nn = modelos_knn[categoria]

    #     # Si no hay suficientes streamers, devolver los que hay sin usar KNN
    #     if X_categoria.shape[0] < n:
    #         return df_categoria_cluster_top[['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank', 'cluster']]
        
    #     centroide = X_categoria.mean().values.reshape(1, -1)
    #     distancias, indices = nn.kneighbors(centroide)
        
    #     # Convertir índices de KNN en índices reales del DataFrame
    #     indices_originales = df_categoria_cluster_top.iloc[indices[0]].index

    #     columnas_validas = ['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank', 'cluster']

    #     recomendaciones = df.loc[indices_originales, columnas_validas]  # Usamos df completo para evitar errores

    #     recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer')
    #     recomendaciones = recomendaciones.sort_values(by=['rank'], ascending=True)  # Ordenar por ranking

    #     return recomendaciones
    # else:
    X_categoria = df_categoria[num_cols].reset_index(drop=True)  # Resetear índices

    nn = modelos_knn[categoria]
    centroide = X_categoria.mean().values.reshape(1, -1)
    distancias, indices = nn.kneighbors(centroide)

    columnas_validas = ['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank']

    # Acceder con iloc para evitar problemas de índice
    recomendaciones = df_categoria.iloc[indices[0]][columnas_validas]

    recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer', keep='first')
    recomendaciones = recomendaciones.sort_values(by=['rank'], ascending=False)

    return recomendaciones
    

def recomendar_streamers_por_categoria_avanzado(categoria: str, df: pd.DataFrame, num_cols, n=5):
    if categoria not in modelos_knn:
        return pd.DataFrame()
    
    df_categoria = df[df['categoria'] == categoria]
    if df_categoria.empty:
        return pd.DataFrame()
    
    # Identificar el clúster del streamer con mejor ranking en la categoría
    cluster_top = df_categoria.sort_values(by=['rank'], ascending=False)['cluster'].iloc[0]
    
    # Filtrar por el clúster top
    df_categoria_cluster = df_categoria[df_categoria['cluster'] == cluster_top]

    # 🚀 Filtrar solo los streamers con rank alto dentro del clúster
    df_categoria_cluster_top = df_categoria_cluster.sort_values(by=['rank'], ascending=False).head(10)

    # Si hay pocos streamers en el clúster, usar toda la categoría
    if df_categoria_cluster_top.shape[0] < 5:
        df_categoria_cluster_top = df_categoria_cluster

    # Aplicar KNN dentro del clúster
    X_categoria = df_categoria_cluster_top[num_cols].reset_index(drop=True)  # Resetear índices para evitar problemas
    nn = modelos_knn[categoria]

    # Si no hay suficientes streamers, devolver los que hay sin usar KNN
    if X_categoria.shape[0] < n:
        return df_categoria_cluster_top[['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank', 'cluster']]

    centroide = X_categoria.mean().values.reshape(1, -1)
    distancias, indices = nn.kneighbors(centroide)

    # Convertir índices de KNN en índices reales del DataFrame
    indices_originales = df_categoria_cluster_top.iloc[indices[0]].index

    columnas_validas = ['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank', 'cluster']

    recomendaciones = df.loc[indices_originales, columnas_validas]  # Usamos df completo para evitar errores

    recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer')
    recomendaciones = recomendaciones.sort_values(by=['rank'], ascending=True)  # Ordenar por ranking

    return recomendaciones




# Interfaz Streamlit
st.title("Recomendador de Streamers por Categoría")
st.write("Este recomendador utiliza un modelo de K-Nearest Neighbors para encontrar streamers similares a los de una categoría seleccionada.")

# Lista de categorías únicas
categorias_unicas = df['categoria'].drop_duplicates().sort_values().tolist()
categoria_seleccionada = st.selectbox("Seleccione una categoría:", categorias_unicas)

radio = st.radio("Elige modelo:", ('Basico', 'Avanzado'))

if st.button("Recomendar"):
    if radio == 'Basico':
        # Cargar modelos KNN entrenados
        modelos_knn, num_cols, scaler = joblib.load("modelos/modelo_basico.pkl")
        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ponderar columnas clave antes de normalizar
        df['rank'] *= 4  # Dar más peso al ranking
        df['total_followers'] *= 2  # Más peso a seguidores
        df['total_views'] *= 2  # Más peso a vistas totales

        df[num_cols] = scaler.fit_transform(df[num_cols])
        resultados = recomendar_streamers_por_categoria(categoria_seleccionada, df, radio, num_cols)
    else:
        # Cargar modelos KNN entrenados
        modelos_knn, num_cols, kmeans, scaler, le = joblib.load("modelos/modelo_avanzado.pkl")
        num_cols.remove('categoria_codificada')
        num_cols.remove('cluster')
        # Preprocesamiento de datos
        df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['categoria_codificada'] = le.fit_transform(df['categoria'])

        num_cols.append('categoria_codificada')

        # Ponderar columnas clave antes de normalizar
        df['rank'] *= 4  # Dar más peso al ranking
        df['total_followers'] *= 2  # Más peso a seguidores
        df['total_views'] *= 2  # Más peso a vistas totales

        df[num_cols] = scaler.fit_transform(df[num_cols])

        df["cluster"] = kmeans.predict(df[num_cols])

        num_cols.append('cluster')

        resultados = recomendar_streamers_por_categoria_avanzado(categoria_seleccionada, df, num_cols)
    if not resultados.empty:
        st.write(f"### Streamers Recomendados en la Categoría: {categoria_seleccionada}")
        st.dataframe(
    resultados.drop(columns=["id_streamer"])
    .rename(columns={'nombre': 'Nombre', 'categoria': 'Categoría', 'total_followers': 'Seguidores', 'total_views': 'Vistas Totales', 'rank': 'Ranking'})
    .reset_index(drop=True)
    .head(5))

    else:
        st.write("No se encontraron recomendaciones para la categoría seleccionada.")




# import streamlit as st
# import pandas as pd
# import numpy as np
# from pymongo import MongoClient
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# from src import soporte_bbdd_mongo as sbm

# # Conectar a MongoDB
# client, db = sbm.conectar_mongo()

# # Cargar datos
# historico = pd.DataFrame(list(db.historico_streamers_total.find()))
# categorias = pd.DataFrame(list(db.categorias_streameadas.find()))
# streamers = pd.DataFrame(list(db.streamers.find()))  # Cargar ranking

# client.close()

# # Unir datasets
# historico = historico.drop(columns=['_id'])
# categorias = categorias.drop(columns=['_id'])
# streamers = streamers.drop(columns=['_id'])
# df = historico[["id_streamer", "avgviewers", "peakviewers", "hoursstreamed"]].merge(categorias, on='id_streamer', how='left')
# df = df.merge(streamers[['id_streamer', 'rank', 'all_time_peak_viewers', 'total_followers', 'total_views']], on='id_streamer', how='left')  # Agregar ranking

# # df["rank"] = df["rank"].fillna(df["rank"].median())
# df["rank"] = df["rank"].max() - df["rank"]

# # Interfaz Streamlit
# st.title("Recomendador de Streamers por Categoría")
# st.write("Este recomendador utiliza un modelo de K-Nearest Neighbors para encontrar streamers similares a los de una categoría seleccionada.")

# # Lista de categorías únicas
# categorias_unicas = df['categoria'].drop_duplicates().sort_values().tolist()
# categoria_seleccionada = st.selectbox("Seleccione una categoría:", categorias_unicas)
# st.radio("Elige modelo:", ('Basico', 'Avanzado'))
# st.radio("¿Desea filtrar por cluster?", ('Sí', 'No'))

# # Cargar modelos KNN entrenados
# modelos_knn, num_cols, kmeans, scaler, le = joblib.load("../streamlit/modelos/modelo_lastversion.pkl")

# num_cols.remove('categoria_codificada')
# num_cols.remove('cluster')

# # Preprocesamiento de datos
# df.replace(r'(?<!.)-(?!.)', np.nan, regex=True, inplace=True)
# for col in num_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
# # df[num_cols] = df[num_cols].fillna(df[num_cols].median())
# # scaler = MinMaxScaler()

# df['categoria_codificada'] = le.fit_transform(df['categoria'])

# num_cols.append('categoria_codificada')

# # Ponderar columnas clave antes de normalizar
# df['rank'] *= 5  # Dar más peso al ranking
# df['total_followers'] *= 3  # Más peso a seguidores
# df['total_views'] *= 3  # Más peso a vistas totales

# df[num_cols] = scaler.fit_transform(df[num_cols])

# df["cluster"] = kmeans.predict(df[num_cols])

# num_cols.append('cluster')

# def recomendar_streamers_por_categoria(categoria: str, df:pd.DataFrame, n=5):
    
#     if categoria not in modelos_knn:
#         return pd.DataFrame()
    
#     df_categoria = df[df['categoria'] == categoria]
#     if df_categoria.empty:
#         return pd.DataFrame()
#     elif df_categoria.shape[0] < 5:
#         return df_categoria
    
#     X_categoria = df_categoria[num_cols]
#     nn = modelos_knn[categoria]

    
    
#     centroide = X_categoria.mean().values.reshape(1, -1)
#     distancias, indices = nn.kneighbors(centroide)
    
#     columnas_validas = ['id_streamer', 'nombre', 'total_followers', 'total_views', 'categoria', 'rank']
    
#     recomendaciones = df_categoria.iloc[indices[0]][columnas_validas]
    
#     recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer')
#     recomendaciones = recomendaciones.sort_values(by=['rank'], ascending=False)  # Ordenar por ranking
    
#     return recomendaciones

# if st.button("Recomendar"):
#     resultados = recomendar_streamers_por_categoria(categoria_seleccionada, df)
#     if not resultados.empty:
#         st.write(f"### Streamers Recomendados en la Categoría: {categoria_seleccionada}")
#         st.dataframe(
#     resultados.drop(columns=["id_streamer"])
#     .rename(columns={'nombre': 'Nombre', 'categoria': 'Categoría', 'total_followers': 'Seguidores', 'total_views': 'Vistas Totales', 'rank': 'Ranking'})
#     .reset_index(drop=True)
#     .head(5)
# )

#     else:
#         st.write("No se encontraron recomendaciones para la categoría seleccionada.")
