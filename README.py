#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
README - Análisis de la Relación entre PIB e IVA en Colombia (2000-2024)

Este archivo contiene instrucciones para ejecutar el análisis completo
de la relación entre el PIB y la recaudación de IVA en Colombia,
incluyendo análisis de series temporales y modelado con redes neuronales.

Autor: Manus AI
Fecha: Mayo 2025
"""

# Estructura del Proyecto
# -----------------------
# 
# El proyecto está organizado en las siguientes carpetas:
# 
# - /codigo: Scripts de Python para análisis y modelado
# - /datos: Archivos CSV con datos de PIB e IVA
# - /visualizaciones: Gráficos y visualizaciones generadas
# - /informe: Informe académico en formato APA 7
# - /resultados: Resultados de análisis y métricas (generado durante ejecución)
# - /modelos: Modelos entrenados de redes neuronales (generado durante ejecución)

# Requisitos
# ----------
# 
# Para ejecutar este proyecto se requieren las siguientes bibliotecas de Python:
# 
# - pandas
# - numpy
# - matplotlib
# - seaborn
# - statsmodels
# - scikit-learn
# - tensorflow (para redes neuronales)
# - keras (incluido en tensorflow)
# 
# Puede instalar todas las dependencias con:
# pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow

# Flujo de Ejecución Recomendado
# ------------------------------
# 
# Para reproducir el análisis completo, se recomienda ejecutar los scripts en el siguiente orden:
# 
# 1. obtener_pib_colombia.py - Obtiene y procesa datos del PIB de Colombia
# 2. estandarizar_alinear_series.py - Alinea las series temporales de PIB e IVA
# 3. analisis_series_temporales_depurado.py - Realiza análisis de series temporales y modelado SARIMA
# 4. redes_neuronales_series_temporales.py - Implementa y evalúa modelos de redes neuronales

# Instrucciones para VSCode
# -------------------------
# 
# 1. Abra el proyecto en VSCode
# 2. Asegúrese de tener instalada la extensión de Python
# 3. Configure un entorno virtual (opcional pero recomendado):
#    python -m venv venv
#    source venv/bin/activate  # En Windows: venv\Scripts\activate
# 4. Instale las dependencias:
#    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow
# 5. Ejecute los scripts en el orden recomendado usando el botón de "Run" de VSCode
#    o desde la terminal con: python codigo/nombre_script.py

# Notas Importantes
# ----------------
# 
# - Los scripts están diseñados para ser ejecutados desde la raíz del proyecto
# - Algunos scripts pueden tardar varios minutos en ejecutarse, especialmente el entrenamiento de redes neuronales
# - Los resultados pueden variar ligeramente debido a la naturaleza estocástica del entrenamiento de redes neuronales
# - El informe completo en formato APA 7 se encuentra en /informe/informe_apa7.md

# Para convertir el informe Markdown a PDF
# ---------------------------------------
# 
# Si desea convertir el informe de Markdown a PDF, puede utilizar herramientas como:
# - Pandoc: pandoc -s informe/informe_apa7.md -o informe/informe_apa7.pdf
# - Extensiones de VSCode como "Markdown PDF"
# - Servicios web como "Markdown to PDF"

print("Proyecto de Análisis de Relación PIB-IVA en Colombia (2000-2024)")
print("Para comenzar, ejecute los scripts en el orden recomendado en el README.")
