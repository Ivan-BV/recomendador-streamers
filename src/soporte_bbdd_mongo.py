from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi
import pandas as pd
import numpy as np
import streamlit as st

def conectar_mongo():
    """
    Conecta a la base de datos MongoDB usando variables de entorno.
    
    Returns:
        Cliente MongoDB y base de datos conectada.
    """
    uri: str = st.secrets['security']['MONGO_URI_ATLAS']
    db_name: str = st.secrets['security']['MONGO_DB_NAME']
    client = MongoClient(uri)
    db = client[db_name]
    return client, db

def obtener_datos_historicos_streamer(nombre_streamer: str):
    """
    Obtiene los datos de un streamer específico desde MongoDB.
    
    :param nombre_streamer: Nombre del streamer a buscar.
    :param db_name: Nombre de la base de datos en MongoDB.
    :param collection_name: Nombre de la colección en MongoDB.
    :return: Lista de documentos con los datos del streamer.
    """
    # Conectar con MongoDB
    client, db = conectar_mongo()
    collection_name="historico_streamers"
    collection = db[collection_name]

    # Filtrar por nombre de streamer
    resultados = list(collection.find({"nombre": nombre_streamer}))

    # Cerrar conexión
    client.close()

    return resultados

# Modelo basico
def cargar_ranking_desde_mongo(db: Database):
    """
    Se conecta a la colección 'historico_streamers' y obtiene los registros que
    contienen la información actual de cada streamer (por ejemplo, la información de ranking).
    Si hay varios documentos por streamer, se toma el más reciente (suponiendo que existe un campo 'fecha').
    """
    if db is None:
        client, db = conectar_mongo()
    cursor = db.get_collection("streamers").find()
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
    # Si existe un campo de fecha, seleccionamos el registro más reciente por streamer.
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.sort_values("fecha", ascending=False)
        df = df.groupby("Nombre", as_index=False).first()
    return df

def obtener_datos_para_modelo_basico():
    """
    Extrae los datos necesarios para el modelo básico desde Mongo.
    Se utiliza la colección 'historico_streamers' para obtener la información actual (ranking y demás).
    Se asume que existe el campo "Rank".
    """
    client, db = conectar_mongo()
    df_ranking = cargar_ranking_desde_mongo(db)
    
    if "rank" not in df_ranking.columns:
        raise ValueError("El campo 'Rank' no se encuentra en los datos de 'historico_streamers'.")
        
    df_ranking["clasificacion"] = df_ranking["rank"].apply(clasificar_rank)
    return df_ranking

# Función para clasificar el ranking
def clasificar_rank(rank):
    if rank < 100:
        return "Top"
    elif rank <= 300:
        return "Medio"
    else:
        return "Bajo"

# Modelo avanzado
def cargar_historial_desde_mongo(db=None):
    """
    Carga todos los registros históricos de la colección 'historico_streamers'.
    Se asume que esta colección contiene documentos con datos de diferentes momentos (por ejemplo, con campos Year y Month).
    """
    if db is None:
        db = conectar_mongo()
    cursor = db.historico_streamers.find()
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df

def cargar_categorias_desde_mongo(db=None):
    """
    Carga los datos de la colección 'categorias_streameadas' y retorna un DataFrame.
    """
    if db is None:
        db = conectar_mongo()
    cursor = db.categorias_streameadas.find()
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df
        
def obtener_datos_para_modelo_avanzado():
    """
    Extrae y procesa la información para el modelo avanzado.
    
    Se realiza lo siguiente:
      1. Se carga el ranking actual (desde 'historico_streamers') y se normaliza el campo 'Nombre'.
      2. Se carga el historial completo (desde 'historico_streamers') y se calcula para cada streamer:
         - Promedio de AvgViewers
         - Promedio de HoursStreamed
         - Último valor de Followers (total_followers_hist)
         - Veteranía (años transcurridos desde el primer registro, a partir de Year y Month)
      3. Se carga la colección 'categorias_streameadas' para agregar, por ejemplo, el número de categorías únicas.
      4. Se realiza un merge de la información para construir el DataFrame final para el modelo.
    """
    db = conectar_mongo()
    # Cargar ranking y historial
    df_ranking = cargar_ranking_desde_mongo(db)
    df_historial = cargar_historial_desde_mongo(db)
    
    # Normalizar la columna "Nombre" en ambos DataFrames
    df_ranking["Nombre"] = df_ranking["Nombre"].astype(str).str.strip().str.lower()
    df_historial["Nombre"] = df_historial["Nombre"].astype(str).str.strip().str.lower()
    
    # Clasificar según Rank (del ranking actual)
    def clasificar_rank(rank):
        if rank < 100:
            return "Top"
        elif rank <= 300:
            return "Medio"
        else:
            return "Bajo"
    df_ranking["clasificacion"] = df_ranking["Rank"].apply(clasificar_rank)
    
    # Procesar el historial para obtener estadísticas agregadas
    lista_stats = []
    for nombre, group in df_historial.groupby("Nombre"):
        # Convertir columnas a numérico (si es que vienen como string)
        for col in ["AvgViewers", "HoursStreamed", "Followers"]:
            if col in group.columns:
                group[col] = pd.to_numeric(group[col], errors="coerce")
        
        # Calcular la veteranía usando Year y Month (se asume Month con formato de 3 letras, ej.: "Jan")
        if "Year" in group.columns and "Month" in group.columns:
            try:
                group["fecha"] = pd.to_datetime(group["Year"].astype(str) + " " + group["Month"], format="%Y %b", errors="coerce")
                fecha_inicial = group["fecha"].min()
                hoy = pd.to_datetime("today")
                veterania = (hoy - fecha_inicial).days / 365
                veterania = round(veterania, 2)
            except Exception as e:
                veterania = np.nan
        else:
            veterania = np.nan

        promedio_viewers = round(group["AvgViewers"].mean(), 3) if "AvgViewers" in group.columns else np.nan
        promedio_hours = round(group["HoursStreamed"].mean(), 3) if "HoursStreamed" in group.columns else np.nan
        total_followers_hist = int(group["Followers"].iloc[-1]) if "Followers" in group.columns and not group["Followers"].isnull().all() else np.nan

        lista_stats.append({
            "Nombre": nombre,
            "promedio_viewers": promedio_viewers,
            "promedio_hours": promedio_hours,
            "total_followers_hist": total_followers_hist,
            "veterania": veterania
        })
    df_stats = pd.DataFrame(lista_stats)
    
    # Cargar y procesar los datos de categorías
    df_categorias = cargar_categorias_desde_mongo(db)
    if not df_categorias.empty:
        # Normalizamos el campo "nombre" para que coincida con "Nombre"
        df_categorias["nombre"] = df_categorias["nombre"].astype(str).str.strip().str.lower()
        # Por cada streamer, contamos el número de categorías únicas
        df_cat_agg = df_categorias.groupby("nombre")["categoría"].nunique().reset_index().rename(columns={"nombre": "Nombre", "categoría": "num_categorias"})
    else:
        df_cat_agg = pd.DataFrame()
    
    # Realizamos el merge: se unen los datos actuales (ranking) con las estadísticas históricas
    df_avanzado = pd.merge(df_ranking, df_stats, on="Nombre", how="left")
    # Se agrega la información de categorías (por ejemplo, número de categorías únicas)
    if not df_cat_agg.empty:
        df_avanzado = pd.merge(df_avanzado, df_cat_agg, on="Nombre", how="left")
    
    return df_avanzado


if __name__ == "__main__":
    # Conectar a MongoDB
    client, db = conectar_mongo()
    client.close()
