import pandas as pd
import numpy as np
import streamlit as st
from src import soporte_bbdd_mongo as sbm

# Cargar datos
@st.cache_data(show_spinner="🔄 Cargando datos, un momento...", ttl=21600)
def load_data(_db):
    """
    Carga los datos desde MongoDB y los convierte en DataFrames de Pandas.

    Esta función obtiene los datos de tres colecciones de MongoDB:
    - `historico_streamers_total`: Información histórica de los streamers.
    - `categorias_streameadas`: Categorías en las que han hecho streaming.
    - `streamers`: Datos generales y ranking de los streamers.

    La función usa `st.cache_data` para mejorar el rendimiento almacenando los resultados
    en caché durante 6 horas (`ttl=21600` segundos).

    Args:
        _db: Conexión a la base de datos MongoDB.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - `historico`: DataFrame con información histórica de los streamers.
            - `categorias`: DataFrame con las categorías streameadas por los streamers.
            - `streamers`: DataFrame con información general de los streamers, incluyendo ranking.
    """
    historico = pd.DataFrame(list(_db.historico_streamers_total.find())).drop(columns=["_id"])
    categorias = pd.DataFrame(list(_db.categorias_streameadas.find())).drop(columns=["_id"])
    streamers = pd.DataFrame(list(_db.streamers.find())).drop(columns=["_id"])  # Cargar ranking
    return historico, categorias, streamers


# Cargar modelos desde la base de datos
@st.cache_data(show_spinner="🔄 Iniciando modelos, un momento...", ttl=21600)
def load_models(_db):
    """
    Carga los modelos de recomendación desde MongoDB.

    Esta función obtiene los modelos `modelo_basico` y `modelo_avanzado` de la colección
    `modelos` en MongoDB. Los modelos están almacenados en formato serializado dentro del campo `"modelo"`.

    La función usa `st.cache_data` para evitar recargas innecesarias, almacenando los modelos en caché
    durante 6 horas (`ttl=21600` segundos).

    Args:
        _db: Conexión a la base de datos MongoDB.

    Returns:
        tuple[Any, Any]: 
            - `modelo_basico`: Modelo básico de recomendación.
            - `modelo_avanzado`: Modelo avanzado de recomendación.
    """
    modelo_basico = _db.modelos.find_one({"nombre": "modelo_basico"})["modelo"]
    modelo_avanzado = _db.modelos.find_one({"nombre": "modelo_avanzado"})["modelo"]
    return modelo_basico, modelo_avanzado

# Función para traducir meses de inglés a español
def traducir_mes(mes):
    """
    Traduce el nombre de un mes de inglés a español.

    Esta función toma un nombre de mes en inglés (abreviado) y lo traduce a su 
    equivalente en español. Si el mes no está en la lista de traducción, 
    devuelve el valor original sin cambios.

    Args:
        mes (str): Nombre del mes en inglés (abreviatura de tres letras).

    Returns:
        str: Nombre del mes en español.
    
    Example:
        >>> traducir_mes("Jan")
        'Enero'
        >>> traducir_mes("Aug")
        'Agosto'
        >>> traducir_mes("XYZ")
        'XYZ'
    """
    traduccion = {
        "Jan": "Enero", "Feb": "Febrero", "Mar": "Marzo", "Apr": "Abril",
        "May": "Mayo", "Jun": "Junio", "Jul": "Julio", "Aug": "Agosto",
        "Sep": "Septiembre", "Oct": "Octubre", "Nov": "Noviembre", "Dec": "Diciembre"
    }
    return traduccion.get(mes, mes)  # Si no lo encuentra, deja el original


# Función para crear un enlace HTML a la página de historial de un streamer
def crear_enlace(name):
    """
    Genera un enlace HTML clickeable a la página de historial de un streamer.

    La función devuelve un enlace en HTML que permite redirigir a la página de 
    historial de un streamer. Se abre en una nueva pestaña y mantiene el estilo 
    del texto en color blanco sin subrayado.

    Args:
        name (str): Nombre del streamer.

    Returns:
        str: Enlace HTML con formato clickeable.

    Example:
        >>> crear_enlace("Rubius")
        '<a href="/historico?streamer=Rubius" target="_blank" style="text-decoration: none; color: white;">Rubius</a>'
    """
    return f'<a href="/historico?streamer={name}" target="_blank" style="text-decoration: none; color: white;">{name}</a>'

def mostrar_datos(df: pd.DataFrame):
    """
    Muestra los datos de los streamers con formato en Streamlit.

    Esta función formatea los datos de los streamers, renombra las columnas para mayor claridad, 
    y convierte los nombres en enlaces HTML clickeables que redirigen a la página de historial 
    de cada streamer. Luego, genera una tabla HTML estilizada y la muestra en Streamlit.

    Se incluyen mejoras visuales como:
    - Separadores de miles en valores numéricos.
    - Estilos CSS personalizados para mejorar la apariencia de la tabla.
    - Efecto hover en las filas para mejorar la experiencia visual.

    Args:
        df (pd.DataFrame): DataFrame con los datos de los streamers.

    Returns:
        None: Renderiza la tabla en Streamlit sin devolver un valor.

    Example:
        >>> df = pd.DataFrame({
        ...     "nombre": ["Rubius", "Ibai"],
        ...     "hoursstreamed": [500, 700],
        ...     "all_time_peak_viewers": [100000, 200000],
        ...     "total_followers": [5000000, 8000000],
        ...     "total_views": [100000000, 200000000],
        ...     "rank": [1, 2]
        ... })
        >>> mostrar_datos(df)
        # Muestra la tabla en Streamlit con los datos formateados
    """
    df = df.reindex(columns=["nombre"] + df.columns.drop(["nombre", "id_streamer", "avgviewers", "peakviewers", "categoria", "rank"]).to_list() + ["rank"])
    df = df.rename(columns={
        "nombre": "Nombre",
        "hoursstreamed": "Horas Streameadas",
        "all_time_peak_viewers": "Pico Máx. Espectadores",
        "total_followers": "Total Seguidores",
        "total_views": "Total Visitas",
        "rank": "Ranking Global"
    })
    df["Nombre"] = df["Nombre"].apply(crear_enlace)

    # Estilos CSS para mejorar la apariencia de la tabla en Streamlit
    st.markdown("""
    <style>
        .tabla-custom {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
            text-align: center;
        }
        .tabla-custom th {
            background-color: #097cff;
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
            background-color: #606060;
        }
    </style>
    """, unsafe_allow_html=True)

    # Generar tabla en HTML
    tabla_html = "<table class='tabla-custom'>"
    tabla_html += "<tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr>"

    for _, row in df.iterrows():
        fila = "<tr>"
        for col in df.columns:
            valor = row[col]
            fila += f"<td>{valor:,.0f}</td>" if isinstance(valor, (int, float)) else f"<td>{valor}</td>"
        fila += "</tr>"
        tabla_html += fila

    tabla_html += "</table>"

    st.markdown(tabla_html, unsafe_allow_html=True)


def mostrar_datos_historicos(df: pd.DataFrame):
    """
    Muestra los datos históricos de los streamers con formato en Streamlit.

    Esta función formatea los datos históricos de los streamers, incluyendo:
    - Traducción de los meses de inglés a español.
    - Ordenación de los meses en orden cronológico.
    - Renombrado de las columnas para mejorar la legibilidad.
    - Conversión y formateo de valores numéricos con separadores de miles.
    - Aplicación de estilos CSS personalizados para mejorar la apariencia de la tabla.
    - Presentación de los datos organizados por año con `st.expander()`.

    Los datos se dividen en secciones por año, donde cada año se muestra en un bloque desplegable.

    Args:
        df (pd.DataFrame): DataFrame con los datos históricos de los streamers.

    Returns:
        None: Renderiza la tabla en Streamlit sin devolver un valor.

    Example:
        >>> df = pd.DataFrame({
        ...     "year": [2024, 2024, 2025],
        ...     "month": ["Jan", "Feb", "Mar"],
        ...     "avgviewers": [10000, 12000, 15000],
        ...     "viewersgain": [500, -300, 700],
        ...     "peakviewers": [50000, 60000, 70000],
        ...     "followers": [100000, 110000, 120000]
        ... })
        >>> mostrar_datos_historicos(df)
        # Muestra la tabla en Streamlit con los datos formateados y agrupados por año.
    """
    
    # Traducir los meses al español y ordenar por mes
    df["month"] = df["month"].apply(traducir_mes)
    meses_ordenados = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    df["month"] = pd.Categorical(df["month"], categories=meses_ordenados, ordered=True)

    # Renombrar columnas para mayor claridad
    df = df.rename(columns={
        "year": "Año",
        "month": "Mes",
        "avgviewers": "Promedio de Viewers",
        "viewersgain": "Cambio en Viewers",
        "percentagegainviewers": "Variación % Viewers",
        "peakviewers": "Pico de Viewers",
        "hoursstreamed": "Horas Streameadas",
        "hoursgain": "Cambio Horas Streameadas",
        "percentagegainhours": "Variación % Horas",
        "followers": "Total Seguidores",
        "percentagegainfollowers": "Variación % Seguidores",
        "followersgain": "Cambio en Seguidores",
        "perhour": "Seguidores por Hora"
    })

    # Convertir a números todas las columnas numéricas para evitar errores de formato
    columnas_numericas = ["Promedio de Viewers", "Cambio en Viewers", "Pico de Viewers",
                          "Horas Streameadas", "Cambio Horas Streameadas", "Total Seguidores",
                          "Cambio en Seguidores", "Seguidores por Hora"]
    
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convierte a float, reemplaza errores con NaN

    # Aplicar formato numérico con separadores de miles
    for col in columnas_numericas:
        df[col] = df[col].apply(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notnull(x) else "N/A")

    # Aplicar formato especial a las columnas de porcentaje
    df["Variación % Viewers"] = df["Variación % Viewers"].apply(lambda x: f"{float(x):+.1f}%" if pd.notnull(x) else "N/A")
    df["Variación % Horas"] = df["Variación % Horas"].apply(lambda x: f"{float(x):+.1f}%" if pd.notnull(x) and x != "-" else "N/A")
    df["Variación % Seguidores"] = df["Variación % Seguidores"].apply(lambda x: f"{float(x):+.1f}%" if pd.notnull(x) else "N/A")

    df["Año"] = df["Año"].astype(str)  # Asegurar que "Año" es string para evitar comas

    # Aplicar estilos con HTML en Streamlit
    st.markdown("""
    <style>
        .tabla-custom {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
            text-align: center;
        }
        .tabla-custom th {
            background-color: #097cff;
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
            background-color: #606060;
        }
    </style>
    """, unsafe_allow_html=True)

    # Mostrar cada año en una sección separada
    for año in sorted(df["Año"].unique(), reverse=True):  # Orden descendente
        with st.expander(f"📅 Datos de {año}", expanded=True):  # Expander para cada año
            df_filtrado = df[df["Año"] == año].sort_values("Mes")
            df_filtrado.drop(columns=["Año", "Seguidores por Hora"], inplace=True)

            # Construcción de tabla en HTML
            tabla_html = "<table class='tabla-custom'>"
            tabla_html += "<tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr>"

            for _, row in df_filtrado.iterrows():
                fila = "<tr>"
                for col in df_filtrado.columns:
                    valor = row[col]
                    fila += f"<td>{valor}</td>"
                fila += "</tr>"
                tabla_html += fila

            tabla_html += "</table>"

            st.markdown(tabla_html, unsafe_allow_html=True)

def recomendar_streamers_por_categoria_avanzado(categoria: str, df: pd.DataFrame, modelos_knn_avanzado, num_cols_avanzado, n=5):
    """
    Genera una lista de streamers recomendados en base a una categoría utilizando un modelo KNN avanzado.

    Esta función filtra los streamers por la categoría especificada, ordena los datos por ranking y
    utiliza un modelo KNN (K-Nearest Neighbors) avanzado para identificar los streamers más similares 
    al perfil promedio dentro de la categoría. 

    Si la categoría no tiene streamers o no está en los modelos disponibles, devuelve un DataFrame vacío.

    Args:
        categoria (str): Categoría de streamers para la cual se generarán recomendaciones.
        df (pd.DataFrame): DataFrame con información de los streamers.
        modelos_knn_avanzado (dict): Modelos KNN preentrenados para cada categoría.
        num_cols_avanzado (list): Lista de columnas numéricas utilizadas en el modelo.
        n (int, optional): Número de streamers recomendados a devolver. Por defecto, `n=5`.

    Returns:
        pd.DataFrame: DataFrame con los streamers recomendados, incluyendo:
            - `id_streamer`: ID único del streamer.
            - `nombre`: Nombre del streamer.
            - `categoria`: Categoría en la que se encuentra.
            - `rank`: Posición en el ranking global.

    Example:
        >>> df = pd.DataFrame({
        ...     "id_streamer": [1, 2, 3, 4, 5],
        ...     "nombre": ["Streamer1", "Streamer2", "Streamer3", "Streamer4", "Streamer5"],
        ...     "categoria": ["Just Chatting", "Just Chatting", "Gaming", "Gaming", "Gaming"],
        ...     "rank": [100, 200, 50, 75, 150]
        ... })
        >>> modelos_knn_avanzado = {"Just Chatting": knn_model}
        >>> num_cols_avanzado = ["rank"]
        >>> recomendar_streamers_por_categoria_avanzado("Just Chatting", df, modelos_knn_avanzado, num_cols_avanzado, n=3)
        
        # Devuelve un DataFrame con los 3 mejores streamers recomendados en "Just Chatting".
    """

    # Filtrar los streamers que pertenecen a la categoría solicitada
    df_categoria = df[df['categoria'] == categoria]

    if df_categoria.empty:
        print(f"⚠️ No hay streamers en la categoría {categoria}")
        return pd.DataFrame()

    # 🔹 Ordenar por ranking de forma descendente (ranking más alto primero)
    df_categoria_ordenado = df_categoria.sort_values(by=['rank'], ascending=False)

    if categoria in modelos_knn_avanzado:
        X_categoria = df_categoria[num_cols_avanzado]
        nn = modelos_knn_avanzado[categoria]

        # Obtener el centroide de los streamers en la categoría
        centroide = X_categoria.mean().values.reshape(1, -1)
        distancias, indices = nn.kneighbors(centroide)

        columnas_validas = ['id_streamer', 'nombre', 'categoria', 'rank']

        recomendaciones = pd.DataFrame()

        # Seleccionar los streamers más cercanos al centroide en el espacio numérico
        for indice in indices[0]:
            df_categoria_ordenado = pd.concat([recomendaciones, df_categoria_ordenado[df_categoria_ordenado.index == indice]], axis=0)

    # Seleccionar las columnas finales para la recomendación
    columnas_validas = ['id_streamer', 'nombre', 'categoria', 'rank']
    recomendaciones = df_categoria.sort_values(by=['rank'], ascending=False)[columnas_validas]

    return recomendaciones

def recomendar_streamers_por_categoria(categoria, df, modelos_knn_basico, num_cols_basico, n=5):
    """
    Genera una lista de streamers recomendados en base a una categoría utilizando un modelo KNN básico.

    La función filtra los streamers por la categoría especificada y utiliza un modelo KNN (K-Nearest Neighbors)
    para identificar los streamers más similares al perfil promedio dentro de la categoría. 

    Si la categoría no tiene streamers o no está en los modelos disponibles, devuelve un DataFrame vacío.

    Args:
        categoria (str): Categoría de streamers para la cual se generarán recomendaciones.
        df (pd.DataFrame): DataFrame con información de los streamers.
        modelos_knn_basico (dict): Modelos KNN preentrenados para cada categoría.
        num_cols_basico (list): Lista de columnas numéricas utilizadas en el modelo.
        n (int, optional): Número de streamers recomendados a devolver. Por defecto, `n=5`.

    Returns:
        pd.DataFrame: DataFrame con los streamers recomendados, incluyendo:
            - `id_streamer`: ID único del streamer.
            - `nombre`: Nombre del streamer.
            - `categoria`: Categoría en la que se encuentra.
            - `rank`: Posición en el ranking global.

    Example:
        >>> df = pd.DataFrame({
        ...     "id_streamer": [1, 2, 3, 4, 5],
        ...     "nombre": ["Streamer1", "Streamer2", "Streamer3", "Streamer4", "Streamer5"],
        ...     "categoria": ["Just Chatting", "Just Chatting", "Gaming", "Gaming", "Gaming"],
        ...     "rank": [100, 200, 50, 75, 150]
        ... })
        >>> modelos_knn_basico = {"Just Chatting": knn_model}
        >>> num_cols_basico = ["rank"]
        >>> recomendar_streamers_por_categoria("Just Chatting", df, modelos_knn_basico, num_cols_basico, n=3)
        
        # Devuelve un DataFrame con los 3 mejores streamers recomendados en "Just Chatting".
    """
    if categoria not in modelos_knn_basico:
        return pd.DataFrame()
    
    df_categoria = df[df['categoria'] == categoria]
    if df_categoria.empty:
        return pd.DataFrame()
    
    X_categoria = df_categoria[num_cols_basico]
    nn = modelos_knn_basico[categoria]
    
    # Calcular el centroide de los streamers dentro de la categoría
    centroide = X_categoria.mean().values.reshape(1, -1)
    distancias, indices = nn.kneighbors(centroide)
    
    columnas_validas = ['id_streamer', 'nombre', 'categoria', 'rank']
    
    # Obtener las recomendaciones según la distancia en el espacio de características
    recomendaciones = df_categoria.iloc[indices[0]][columnas_validas]
    
    # Eliminar duplicados y ordenar por ranking descendente
    recomendaciones = recomendaciones.drop_duplicates(subset='id_streamer')
    recomendaciones = recomendaciones.sort_values(by='rank', ascending=False)
    
    return recomendaciones

def preprocesar_datos(historico: pd.DataFrame, categorias: pd.DataFrame, streamers: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza y fusión de datos de los streamers para su posterior análisis o uso en modelos.

    La función combina los datos históricos, las categorías y la información general de los streamers en un único
    DataFrame, aplicando filtros y ajustando el ranking global.

    - Se eliminan algunas categorías no relevantes ("18+", "171", "2XKO", "1v1.LOL").
    - Se ajusta el ranking global de los streamers invirtiendo su valor (ranking más bajo es mejor).

    Args:
        historico (pd.DataFrame): DataFrame con datos históricos de los streamers.
        categorias (pd.DataFrame): DataFrame con las categorías en las que ha streameado cada streamer.
        streamers (pd.DataFrame): DataFrame con información general y ranking de los streamers.

    Returns:
        pd.DataFrame: DataFrame procesado con las siguientes columnas:
            - `id_streamer`: ID único del streamer.
            - `avgviewers`: Promedio de viewers.
            - `peakviewers`: Pico de viewers.
            - `hoursstreamed`: Horas streameadas.
            - `categoria`: Categoría en la que ha streameado.
            - `rank`: Ranking global ajustado.
            - `all_time_peak_viewers`: Pico histórico de viewers.
            - `total_followers`: Total de seguidores.
            - `total_views`: Total de visitas.

    Example:
        >>> historico = pd.DataFrame({
        ...     "id_streamer": [1, 2, 3],
        ...     "avgviewers": [1000, 1500, 2000],
        ...     "peakviewers": [5000, 7000, 9000],
        ...     "hoursstreamed": [100, 200, 300]
        ... })
        >>> categorias = pd.DataFrame({
        ...     "id_streamer": [1, 2, 3],
        ...     "categoria": ["Just Chatting", "Gaming", "IRL"]
        ... })
        >>> streamers = pd.DataFrame({
        ...     "id_streamer": [1, 2, 3],
        ...     "rank": [50, 30, 10],
        ...     "all_time_peak_viewers": [10000, 20000, 30000],
        ...     "total_followers": [100000, 200000, 300000],
        ...     "total_views": [5000000, 10000000, 15000000]
        ... })
        >>> preprocesar_datos(historico, categorias, streamers)
        
        # Devuelve un DataFrame con los datos fusionados y limpiados.
    """
    # Unir los datos históricos con las categorías
    df = historico[["id_streamer", "avgviewers", "peakviewers", "hoursstreamed"]].merge(categorias, on='id_streamer', how='left')

    # Unir con la información general de los streamers
    df = df.merge(streamers[['id_streamer', 'rank', 'all_time_peak_viewers', 'total_followers', 'total_views']], on='id_streamer', how='left')

    # Filtrar categorías no relevantes
    df = df[df['categoria'] != "18+"]
    df = df[df['categoria'] != "171"]
    df = df[df['categoria'] != "2XKO"]
    df = df[df['categoria'] != "1v1.LOL"]

    # Ajustar el ranking global de los streamers invirtiendo su valor (ranking más bajo es mejor)
    df["rank"] = df["rank"].max() + df["rank"].min() - df["rank"]
    df["rank"] = df["rank"].astype(np.int32)

    return df

if __name__ == "__main__":
    load_data()