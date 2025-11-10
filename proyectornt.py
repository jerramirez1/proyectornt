import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Recomendador Turístico Quindío",
    page_icon="🌴",
    layout="wide"
)

# Función para cargar y procesar datos
@st.cache_data
def cargar_datos():
    try:
        # Intenta cargar el archivo CSV
        data = pd.read_csv('Registro_Nacional_de_Turismo_-_RNT_20251016.csv')
        
        # Procesamiento de datos
        data["RAZON_SOCIAL_ESTABLECIMIENTO"].fillna("DESCONOCIDO", inplace=True)
        data["NUMERO_DE_EMPLEADOS"] = pd.to_numeric(data["NUMERO_DE_EMPLEADOS"], errors="coerce")
        data["NUMERO_DE_CAMAS"] = pd.to_numeric(data["NUMERO_DE_CAMAS"], errors="coerce")
        data["NUMERO_DE_HABITACIONES"] = pd.to_numeric(data["NUMERO_DE_HABITACIONES"], errors="coerce")
        
        # Filtrar por Quindío
        df_filtrado_quindio = data[data["DEPARTAMENTO"] == "QUINDIO"].copy()
        
        # Estandarizar nombres de columnas
        df_filtrado_quindio.columns = df_filtrado_quindio.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Reemplazar nulos y limpiar texto
        df_filtrado_quindio.fillna("No disponible", inplace=True)
        
        # Eliminar duplicados
        df_filtrado_quindio.drop_duplicates(inplace=True)
        
        return df_filtrado_quindio
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para calcular similitud
@st.cache_data
def calcular_similitud(df):
    df["info"] = df["categoria"].astype(str) + " " + df["municipio"].astype(str)
    vectorizer = CountVectorizer()
    matriz = vectorizer.fit_transform(df["info"])
    similaridad = cosine_similarity(matriz, matriz)
    return similaridad

# Función de recomendación
def recomendar_establecimientos(nombre, df, similaridad, n=5):
    try:
        idx = df[df["razon_social"].str.contains(nombre, case=False, na=False)].index[0]
        scores = list(enumerate(similaridad[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        indices = [i[0] for i in scores[1:n+1]]
        recomendados = df.iloc[indices][["razon_social", "categoria", "municipio"]]
        return recomendados
    except IndexError:
        return None

# Interfaz principal
def main():
    st.title("🌴 Recomendador Turístico - Quindío")
    st.markdown("Sistema de recomendación basado en el Registro Nacional de Turismo (RNT)")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df_filtrado_quindio = cargar_datos()
    
    if df_filtrado_quindio is None:
        st.warning("⚠️ No se pudieron cargar los datos. Asegúrate de que el archivo CSV esté en el directorio correcto.")
        st.info("📁 El archivo debe llamarse: `Registro_Nacional_de_Turismo_-_RNT_20251016.csv`")
        return
    
    # Calcular similitud
    with st.spinner("Calculando similitudes..."):
        similaridad = calcular_similitud(df_filtrado_quindio)
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["🔍 Recomendador", "📊 Análisis", "📈 Estadísticas"])
    
    with tab1:
        st.header("Buscador de Recomendaciones")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            nombre = st.text_input("Ingrese el nombre o parte del nombre del establecimiento:")
        with col2:
            n = st.slider("Número de recomendaciones", 1, 10, 5)
        
        if st.button("🔍 Buscar Recomendaciones", type="primary"):
            if nombre:
                resultado = recomendar_establecimientos(nombre, df_filtrado_quindio, similaridad, n)
                if resultado is not None:
                    st.success(f"✅ Se encontraron {len(resultado)} recomendaciones similares:")
                    st.dataframe(resultado, use_container_width=True)
                else:
                    st.warning("⚠️ No se encontró el establecimiento. Intenta con otro nombre.")
            else:
                st.info("ℹ️ Por favor, ingresa un nombre para buscar.")
    
    with tab2:
        st.header("Análisis del Sector Turístico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Categorías más frecuentes")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df_filtrado_quindio["categoria"].value_counts().head(10).plot(
                kind="bar", color="teal", ax=ax1
            )
            ax1.set_xlabel("Categoría")
            ax1.set_ylabel("Número de registros")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Municipios con más establecimientos")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df_filtrado_quindio["municipio"].value_counts().head(10).plot(
                kind="barh", color="orange", ax=ax2
            )
            ax2.set_xlabel("Cantidad de registros")
            ax2.set_ylabel("Municipio")
            plt.tight_layout()
            st.pyplot(fig2)
    
    with tab3:
        st.header("Estadísticas Descriptivas")
        
        # Métricas generales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", len(df_filtrado_quindio))
        with col2:
            st.metric("Municipios", df_filtrado_quindio["municipio"].nunique())
        with col3:
            st.metric("Categorías", df_filtrado_quindio["categoria"].nunique())
        
        st.subheader("Estadísticas de Variables Numéricas")
        cols_numericas = ['numero_de_habitaciones', 'numero_de_camas', 'numero_de_empleados']
        
        if all(col in df_filtrado_quindio.columns for col in cols_numericas):
            descriptive_stats = pd.DataFrame({
                'Media': df_filtrado_quindio[cols_numericas].mean(),
                'Mediana': df_filtrado_quindio[cols_numericas].median(),
                'Desviación Estándar': df_filtrado_quindio[cols_numericas].std(),
                'Mínimo': df_filtrado_quindio[cols_numericas].min(),
                'Máximo': df_filtrado_quindio[cols_numericas].max()
            })
            st.dataframe(descriptive_stats, use_container_width=True)
        
        st.subheader("Distribución por Municipio")
        st.bar_chart(df_filtrado_quindio["municipio"].value_counts())
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🏁 <b>Conclusiones:</b></p>
            <p>Sistema de recomendación basado en similitud de texto entre categoría y municipio.</p>
            <p>Herramienta para orientar turistas y analizar la oferta del sector en Quindío.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
