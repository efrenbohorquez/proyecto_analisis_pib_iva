# Análisis de la Relación entre PIB e IVA en Colombia

Este proyecto analiza la relación entre el Producto Interno Bruto (PIB) y la recaudación del Impuesto al Valor Agregado (IVA) en Colombia utilizando datos históricos desde el año 2000 hasta 2024.

## Descripción

El objetivo principal es modelar y predecir estas series temporales, así como entender la dinámica entre ellas. Se utilizan diversas técnicas de análisis de series temporales, incluyendo modelos SARIMA y redes neuronales (RNN).

Los resultados y visualizaciones se presentan en una aplicación interactiva desarrollada con Streamlit.

## Estructura del Proyecto

*   `codigo/`: Contiene todos los scripts de Python para la obtención, procesamiento, análisis de datos y la aplicación Streamlit.
    *   `obtener_pib_colombia.py`: Script para descargar y procesar datos del PIB de Colombia.
    *   `estandarizar_alinear_series.py`: Script para cargar datos del IVA, estandarizarlos y alinearlos con los datos del PIB.
    *   `analisis_series_temporales_depurado.py`: Script para realizar el análisis de series temporales (descomposición, SARIMA, causalidad de Granger).
    *   `redes_neuronales_series_temporales.py`: Script para implementar modelos de redes neuronales para la predicción.
    *   `app_streamlit.py`: Aplicación Streamlit para visualizar los resultados.
*   `datos/`: Almacena los archivos de datos generados y utilizados en el análisis (ej. `pib_mensual_colombia.csv`, `iva_pib_alineado.csv`).
*   `visualizaciones/`: Contiene las gráficas generadas durante el análisis (ej. `series_alineadas.png`, `pronostico_sarima.png`).
*   `resultados/`: Guarda los resultados numéricos y textuales de los análisis (ej. `metricas_sarima.txt`, `correlacion.txt`).
*   `modelos/`: Almacena los modelos entrenados (ej. `modelo_sarima_pib.pkl`, `modelo_rnn.h5`).
*   `requirements.txt`: Lista de dependencias de Python necesarias para ejecutar el proyecto.
*   `README.md`: Este archivo.

## Cómo Ejecutar

1.  **Clonar el repositorio (si aplica):**
    ```bash
    git clone https://github.com/efrenbohorquez/proyecto_analisis_pib_iva.git
    cd proyecto_analisis_pib_iva
    ```
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv .venv
    # Activar el entorno (Windows)
    .venv\Scripts\activate
    # Activar el entorno (Linux/macOS)
    # source .venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecutar los scripts de análisis (en orden):**
    Los scripts están diseñados para ser ejecutados en secuencia o pueden ser invocados por la aplicación Streamlit si está configurada para ello.
    *   `python codigo/obtener_pib_colombia.py`
    *   `python codigo/estandarizar_alinear_series.py`
    *   `python codigo/analisis_series_temporales_depurado.py`
    *   `python codigo/redes_neuronales_series_temporales.py`

5.  **Ejecutar la aplicación Streamlit:**
    ```bash
    streamlit run codigo/app_streamlit.py
    ```
    Esto abrirá la aplicación en tu navegador web.

## Despliegue en Streamlit Community Cloud

Esta aplicación está (o puede ser) desplegada usando Streamlit Community Cloud.
*   **Repositorio:** `https://github.com/efrenbohorquez/proyecto_analisis_pib_iva`
*   **Archivo principal:** `codigo/app_streamlit.py`

## Datos

*   **IVA:** Datos de recaudación mensual del IVA en Colombia. Fuente original: [Incluir fuente si se conoce, ej. DIAN]. Archivo: `D:\Downloads\SERIES DE TIEMPO 2025\series iva\data_iva.xlsx` (ruta local original).
*   **PIB:** Datos del Producto Interno Bruto de Colombia (anual, trimestral, mensual). Fuente: Banco Mundial y DANE, procesados por el script `obtener_pib_colombia.py`.

## Autor

  equipo serie de tiempo
  

---

Puedes personalizar este `README.md` añadiendo más detalles sobre la metodología, los resultados clave, o cualquier otra información que consideres relevante.
