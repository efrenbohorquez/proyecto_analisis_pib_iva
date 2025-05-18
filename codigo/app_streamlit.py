import streamlit as st
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configuración de estilo similar a los scripts originales
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 10 # Ajustado para mejor visualización en Streamlit
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.figsize'] = (8, 5) # Ajustado para Streamlit

# Funciones para formatear ejes (copiadas de los scripts)
def formato_miles_millones(x, pos):
    return f'{x*1e-9:.1f}'

def formato_millones(x, pos):
    return f'{x*1e-6:.1f}'

# Función para cargar y mostrar imágenes
def mostrar_imagen(ruta_imagen, caption=""):
    if os.path.exists(ruta_imagen):
        st.image(ruta_imagen, caption=caption, use_container_width=True) # Parámetro actualizado
    else:
        st.warning(f"No se encontró la imagen: {ruta_imagen}")

# Función para leer contenido de archivos de texto
def leer_archivo_texto(ruta_archivo):
    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(ruta_archivo, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                return f"Error al leer el archivo {ruta_archivo} con UTF-8 y Latin-1: {e}"
    else:
        return f"No se encontró el archivo: {ruta_archivo}"

# --- Configuración de la Página ---
st.set_page_config(page_title="Análisis PIB e IVA Colombia", layout="wide")

st.title("Análisis de la Relación entre PIB e IVA en Colombia (2000-2024)")
st.markdown("**Resultados generados por los scripts de análisis**")

# --- Rutas a los archivos ---
RUTA_DATOS = "../datos/"
RUTA_VISUALIZACIONES = "../visualizaciones/"
RUTA_RESULTADOS = "../resultados/"
RUTA_MODELOS = "../modelos/"

# --- Sección de Carga y Alineación de Datos ---
st.header("1. Carga y Alineación de Datos")

st.subheader("Datos Crudos")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "iva_datos_crudos.png"), "Recaudación de IVA (Datos Crudos)")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "pib_mensual_datos_crudos.png"), "PIB Mensual (Datos Crudos)")

st.subheader("Series Alineadas")
df_alineado_path = os.path.join(RUTA_DATOS, "iva_pib_alineado.csv")
if os.path.exists(df_alineado_path):
    df_alineado = pd.read_csv(df_alineado_path, parse_dates=['fecha_estandar'], index_col='fecha_estandar')
    st.write("Vista previa de los datos alineados (IVA y PIB):")
    st.dataframe(df_alineado.head())
    mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "series_alineadas.png"), "IVA y PIB (Series Alineadas)")
else:
    st.error(f"No se encontró el archivo de datos alineados: {df_alineado_path}")


# --- Sección de Análisis Exploratorio de Datos (EDA) ---
st.header("2. Análisis Exploratorio de Datos (EDA)")

st.subheader("Estadísticas Descriptivas")
st.text_area("Estadísticas IVA y PIB", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "estadisticas_descriptivas.txt")), height=300)

st.subheader("Visualizaciones del EDA")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "series_temporales_iva_pib.png"), "Series Temporales de IVA y PIB")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "dispersion_iva_pib.png"), "Gráfico de Dispersión IVA vs PIB")
st.text("Correlación entre IVA y PIB:")
st.code(leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "correlacion_iva_pib.txt")))
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "variacion_porcentual_anual.png"), "Variación Porcentual Anual: IVA vs PIB")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "estacionalidad_mensual_iva.png"), "Estacionalidad Mensual del IVA")

# --- Sección de Pruebas de Estacionariedad y Cointegración ---
st.header("3. Pruebas de Estacionariedad y Cointegración")

st.subheader("Resultados de las Pruebas")
st.text_area("Pruebas de Estacionariedad", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "pruebas_estacionariedad.txt")), height=400)
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "series_diferenciadas.png"), "Series Diferenciadas (IVA y PIB)")
st.text_area("Prueba de Cointegración", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "prueba_cointegracion.txt")), height=200)

# --- Sección de Descomposición de Series ---
st.header("4. Descomposición de Series Temporales")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "descomposicion_iva.png"), "Descomposición de la Serie Temporal del IVA")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "descomposicion_pib.png"), "Descomposición de la Serie Temporal del PIB")
df_componentes_path = os.path.join(RUTA_RESULTADOS, "componentes_series.csv")
if os.path.exists(df_componentes_path):
    st.write("Componentes de las series (tendencia, estacionalidad, residuo):")
    df_componentes = pd.read_csv(df_componentes_path, parse_dates=['fecha_estandar'], index_col='fecha_estandar')
    st.dataframe(df_componentes.head())
else:
    st.warning(f"No se encontró el archivo de componentes: {df_componentes_path}")

# --- Sección de Modelado SARIMA ---
st.header("5. Modelado SARIMA (para IVA)")
st.subheader("Análisis ACF y PACF")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "acf_pacf_iva.png"), "Funciones de Autocorrelación (ACF y PACF) para IVA")

st.subheader("Resultados del Modelo SARIMA")
st.text_area("Resumen del Modelo SARIMA", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "resumen_modelo_sarima_iva.txt")), height=500)
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "modelo_sarima_iva.png"), "Modelo SARIMA para IVA: Ajuste y Pronóstico")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "diagnostico_residuos_sarima.png"), "Diagnóstico de Residuos del Modelo SARIMA")
st.text_area("Métricas del Modelo SARIMA", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "metricas_modelo_sarima_iva.txt")), height=150)

st.subheader("Prueba de Causalidad de Granger")
st.text_area("Resultados Causalidad de Granger", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "causalidad_granger.txt")), height=300)

# --- Sección de Modelado con Redes Neuronales ---
st.header("6. Modelado con Redes Neuronales (LSTM y GRU para IVA)")
st.subheader("Resultados de los Modelos RNN")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "historia_entrenamiento_rnn.png"), "Historia de Entrenamiento (Loss) de Modelos RNN")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "predicciones_rnn.png"), "Predicciones de Modelos RNN vs Valores Reales")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "pronostico_futuro_rnn.png"), "Pronóstico Futuro con Modelos RNN")

st.text_area("Métricas de los Modelos de Redes Neuronales", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "metricas_redes_neuronales.txt")), height=200)

st.subheader("Documentación de Redes Neuronales")
doc_rnn_path = os.path.join(RUTA_RESULTADOS, "documentacion_redes_neuronales.md")
if os.path.exists(doc_rnn_path):
    st.markdown(leer_archivo_texto(doc_rnn_path))
else:
    st.warning(f"No se encontró el archivo de documentación RNN: {doc_rnn_path}")

# --- Sección de Análisis Comparativo ---
st.header("7. Análisis Comparativo de Modelos")
mostrar_imagen(os.path.join(RUTA_VISUALIZACIONES, "comparacion_modelos.png"), "Comparación de Modelos (SARIMA, LSTM, GRU)")
st.text_area("Conclusiones del Análisis Comparativo", leer_archivo_texto(os.path.join(RUTA_RESULTADOS, "analisis_comparativo.txt")), height=300)

# --- Información Adicional ---
st.sidebar.header("Sobre el Proyecto")
st.sidebar.info("Este dashboard presenta los resultados del análisis de la relación entre el Producto Interno Bruto (PIB) y la recaudación del Impuesto al Valor Agregado (IVA) en Colombia, utilizando datos del periodo 2000-2024.")
st.sidebar.header("Archivos Generados")
st.sidebar.markdown("**Datos:**")
archivos_datos = [f for f in os.listdir(RUTA_DATOS) if os.path.isfile(os.path.join(RUTA_DATOS, f))]
for archivo in archivos_datos:
    st.sidebar.markdown(f"- `{archivo}`")

st.sidebar.markdown("**Resultados (Texto):**")
archivos_resultados_txt = [f for f in os.listdir(RUTA_RESULTADOS) if os.path.isfile(os.path.join(RUTA_RESULTADOS, f)) and f.endswith('.txt')]
for archivo in archivos_resultados_txt:
    st.sidebar.markdown(f"- `{archivo}`")

st.sidebar.markdown("**Modelos:**")
archivos_modelos = [f for f in os.listdir(RUTA_MODELOS) if os.path.isfile(os.path.join(RUTA_MODELOS, f))]
for archivo in archivos_modelos:
    st.sidebar.markdown(f"- `{archivo}`")

st.markdown("--- ")
st.markdown("Fin del reporte.")

