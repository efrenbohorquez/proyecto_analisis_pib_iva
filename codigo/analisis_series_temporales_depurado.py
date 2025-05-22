#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Series Temporales: Relación entre PIB e IVA en Colombia (2000-2024)

Este script realiza un análisis completo de series temporales para estudiar
la relación entre el PIB y la recaudación de IVA en Colombia, incluyendo:
- Análisis exploratorio de datos (EDA)
- Pruebas de estacionariedad
- Descomposición de series temporales
- Modelado ARIMA/SARIMA
- Análisis de cointegración
- Visualizaciones según estilo The Economist

Autor: Manus AI
Fecha: Mayo 2025
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, coint, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import joblib # Para guardar modelos

# Configuración para ignorar advertencias
warnings.filterwarnings('ignore')

# Configuración de visualización al estilo The Economist
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 8)

# Obtener la ruta base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, "..", "datos")
VISUALIZACIONES_DIR = os.path.join(BASE_DIR, "..", "visualizaciones")
RESULTADOS_DIR = os.path.join(BASE_DIR, "..", "resultados")
MODELOS_DIR = os.path.join(BASE_DIR, "..", "modelos")

# Crear directorios si no existen
os.makedirs(VISUALIZACIONES_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Función para formatear valores en miles de millones
def formato_miles_millones(x, pos):
    """Formatea valores en miles de millones con un decimal"""
    return f'{x*1e-9:.1f}'

# Función para formatear valores en millones
def formato_millones(x, pos):
    """Formatea valores en millones con un decimal"""
    return f'{x*1e-6:.1f}'

# Función para cargar datos alineados
def cargar_datos_alineados():
    """
    Carga los datos alineados de IVA y PIB
    """
    print("Cargando datos alineados de IVA y PIB...")
    
    # Cargar datos desde el archivo CSV
    file_path_alineado = os.path.join(DATOS_DIR, "iva_pib_alineado.csv")
    df_conjunto = pd.read_csv(file_path_alineado)
    
    # Convertir columna de fecha a datetime y establecer como índice
    df_conjunto['fecha_estandar'] = pd.to_datetime(df_conjunto['fecha_estandar'])
    df_conjunto.set_index('fecha_estandar', inplace=True)
    
    print(f"Datos alineados cargados: {len(df_conjunto)} registros mensuales")
    print(f"Periodo: {df_conjunto.index.min().strftime('%Y-%m')} a {df_conjunto.index.max().strftime('%Y-%m')}")
    
    return df_conjunto

# Función para realizar análisis exploratorio de datos (EDA)
def realizar_eda(df_conjunto):
    """
    Realiza análisis exploratorio de datos para IVA y PIB
    """
    print("Realizando análisis exploratorio de datos...")
    
    # Estadísticas descriptivas
    stats_iva = df_conjunto['iva'].describe()
    stats_pib = df_conjunto['pib_usd'].describe()
    
    # Guardar estadísticas descriptivas
    with open(os.path.join(RESULTADOS_DIR, 'estadisticas_descriptivas.txt'), 'w') as f:
        f.write("Estadísticas Descriptivas - IVA\n")
        f.write("===============================\n")
        f.write(str(stats_iva))
        f.write("\n\nEstadísticas Descriptivas - PIB\n")
        f.write("===============================\n")
        f.write(str(stats_pib))
    
    # Visualización 1: Series temporales de IVA y PIB
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Gráfico de IVA
    ax1.plot(df_conjunto.index, df_conjunto['iva'], color='#006BA2', linewidth=2)
    ax1.set_title('Recaudación de IVA en Colombia (2000-2024)', fontweight='bold')
    ax1.set_ylabel('IVA (Millones de pesos)')
    ax1.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de PIB
    ax2.plot(df_conjunto.index, df_conjunto['pib_usd'], color='#A2C510', linewidth=2)
    ax2.set_title('PIB de Colombia (2000-2024)', fontweight='bold')
    ax2.set_ylabel('PIB (Miles de millones USD)')
    ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
    ax2.grid(True, alpha=0.3)
    
    # Configuración de eje X
    ax2.set_xlabel('Año')
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_temporales_iva_pib.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Visualización 2: Gráfico de dispersión IVA vs PIB
    fig = plt.figure(figsize=(10, 8))
    ax = sns.regplot(x='pib_usd', y='iva', data=df_conjunto, 
                scatter_kws={'alpha':0.5, 'color':'#006BA2'}, 
                line_kws={'color':'#A2C510', 'linewidth':2})
    plt.title('Relación entre PIB e IVA en Colombia (2000-2024)', fontweight='bold')
    plt.xlabel('PIB (Miles de millones USD)')
    plt.ylabel('IVA (Millones de pesos)')
    ax.xaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
    ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'dispersion_iva_pib.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Visualización 3: Variación porcentual anual
    df_anual = df_conjunto.resample('Y').mean()
    df_anual['iva_var_pct'] = df_anual['iva'].pct_change() * 100
    df_anual['pib_var_pct'] = df_anual['pib_usd'].pct_change() * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_anual.index.year[1:], df_anual['iva_var_pct'][1:], 
           alpha=0.7, color='#006BA2', label='IVA')
    ax.bar(df_anual.index.year[1:], df_anual['pib_var_pct'][1:], 
           alpha=0.7, color='#A2C510', label='PIB', width=0.5)
    ax.set_title('Variación Porcentual Anual: IVA vs PIB (2001-2024)', fontweight='bold')
    ax.set_xlabel('Año')
    ax.set_ylabel('Variación Porcentual (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'variacion_porcentual_anual.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Visualización 4: Estacionalidad mensual del IVA
    df_estacional = df_conjunto.copy()
    df_estacional['mes'] = df_estacional.index.month
    df_estacional['año'] = df_estacional.index.year
    
    estacionalidad = df_estacional.groupby('mes')['iva'].mean()
    
    fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=estacionalidad.index, y=estacionalidad.values, 
                    palette='Blues_d')
    
    # Añadir etiquetas de meses
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax.set_xticklabels(meses)
    
    plt.title('Estacionalidad Mensual de la Recaudación de IVA (2000-2024)', fontweight='bold')
    plt.xlabel('Mes')
    plt.ylabel('IVA Promedio (Millones de pesos)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'estacionalidad_mensual_iva.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Calcular correlación
    correlacion = df_conjunto['iva'].corr(df_conjunto['pib_usd'])
    print(f"Correlación entre IVA y PIB: {correlacion:.4f}")
    
    # Guardar correlación
    with open(os.path.join(RESULTADOS_DIR, 'correlacion_iva_pib.txt'), 'w') as f:
        f.write(f"Correlación entre IVA y PIB: {correlacion:.4f}")
    
    print("Análisis exploratorio completado.")
    return df_conjunto

# Función para realizar pruebas de estacionariedad
def pruebas_estacionariedad(df_conjunto):
    """
    Realiza pruebas de estacionariedad para las series de IVA y PIB
    """
    print("Realizando pruebas de estacionariedad...")
    
    # Función para realizar prueba ADF
    def test_adf(serie, nombre):
        resultado = adfuller(serie.dropna())
        
        print(f"Prueba ADF para {nombre}:")
        print(f"Estadístico ADF: {resultado[0]:.4f}")
        print(f"Valor p: {resultado[1]:.4f}")
        print(f"Valores críticos:")
        for key, value in resultado[4].items():
            print(f"   {key}: {value:.4f}")
        
        if resultado[1] < 0.05:
            print(f"Conclusión: La serie {nombre} es estacionaria (rechaza H0)")
        else:
            print(f"Conclusión: La serie {nombre} no es estacionaria (no rechaza H0)")
        
        return resultado
    
    # Función para realizar prueba KPSS
    def test_kpss(serie, nombre):
        resultado = kpss(serie.dropna())
        
        print(f"\nPrueba KPSS para {nombre}:")
        print(f"Estadístico KPSS: {resultado[0]:.4f}")
        print(f"Valor p: {resultado[1]:.4f}")
        print(f"Valores críticos:")
        for key, value in resultado[3].items():
            print(f"   {key}: {value:.4f}")
        
        if resultado[1] < 0.05:
            print(f"Conclusión: La serie {nombre} no es estacionaria (rechaza H0)")
        else:
            print(f"Conclusión: La serie {nombre} es estacionaria (no rechaza H0)")
        
        return resultado
    
    # Realizar pruebas para IVA
    print("\n" + "="*50)
    print("PRUEBAS DE ESTACIONARIEDAD PARA IVA")
    print("="*50)
    adf_iva = test_adf(df_conjunto['iva'], 'IVA')
    kpss_iva = test_kpss(df_conjunto['iva'], 'IVA')
    
    # Realizar pruebas para PIB
    print("\n" + "="*50)
    print("PRUEBAS DE ESTACIONARIEDAD PARA PIB")
    print("="*50)
    adf_pib = test_adf(df_conjunto['pib_usd'], 'PIB')
    kpss_pib = test_kpss(df_conjunto['pib_usd'], 'PIB')
    
    # Diferenciar series y volver a probar
    df_conjunto['iva_diff'] = df_conjunto['iva'].diff()
    df_conjunto['pib_diff'] = df_conjunto['pib_usd'].diff()
    
    # Eliminar valores NaN después de la diferenciación
    df_diff = df_conjunto.dropna(subset=['iva_diff', 'pib_diff'])
    
    # Visualizar series diferenciadas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Gráfico de IVA diferenciado
    ax1.plot(df_diff.index, df_diff['iva_diff'], color='#006BA2', linewidth=1)
    ax1.set_title('Primera Diferencia de IVA', fontweight='bold')
    ax1.set_ylabel('Δ IVA (Millones de pesos)')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de PIB diferenciado
    ax2.plot(df_diff.index, df_diff['pib_diff'], color='#A2C510', linewidth=1)
    ax2.set_title('Primera Diferencia de PIB', fontweight='bold')
    ax2.set_ylabel('Δ PIB (USD)')
    ax2.grid(True, alpha=0.3)
    
    # Configuración de eje X
    ax2.set_xlabel('Año')
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_diferenciadas.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Pruebas para series diferenciadas
    print("\n" + "="*50)
    print("PRUEBAS DE ESTACIONARIEDAD PARA SERIES DIFERENCIADAS")
    print("="*50)
    adf_iva_diff = test_adf(df_diff['iva_diff'], 'IVA diferenciado')
    adf_pib_diff = test_adf(df_diff['pib_diff'], 'PIB diferenciado')
    
    # Guardar resultados
    with open(os.path.join(RESULTADOS_DIR, 'pruebas_estacionariedad.txt'), 'w') as f:
        f.write("PRUEBAS DE ESTACIONARIEDAD\n")
        f.write("=========================\n\n")
        
        f.write("SERIE IVA ORIGINAL\n")
        f.write(f"Prueba ADF - Estadístico: {adf_iva[0]:.4f}, Valor p: {adf_iva[1]:.4f}\n")
        f.write(f"Prueba KPSS - Estadístico: {kpss_iva[0]:.4f}, Valor p: {kpss_iva[1]:.4f}\n\n")
        
        f.write("SERIE PIB ORIGINAL\n")
        f.write(f"Prueba ADF - Estadístico: {adf_pib[0]:.4f}, Valor p: {adf_pib[1]:.4f}\n")
        f.write(f"Prueba KPSS - Estadístico: {kpss_pib[0]:.4f}, Valor p: {kpss_pib[1]:.4f}\n\n")
        
        f.write("SERIE IVA DIFERENCIADA\n")
        f.write(f"Prueba ADF - Estadístico: {adf_iva_diff[0]:.4f}, Valor p: {adf_iva_diff[1]:.4f}\n\n")
        
        f.write("SERIE PIB DIFERENCIADA\n")
        f.write(f"Prueba ADF - Estadístico: {adf_pib_diff[0]:.4f}, Valor p: {adf_pib_diff[1]:.4f}\n")
    
    # Prueba de cointegración
    print("\n" + "="*50)
    print("PRUEBA DE COINTEGRACIÓN")
    print("="*50)
    
    # Realizar prueba de cointegración de Engle-Granger
    coint_result = coint(df_conjunto['iva'], df_conjunto['pib_usd'])
    
    print(f"Prueba de cointegración de Engle-Granger:")
    print(f"Estadístico: {coint_result[0]:.4f}")
    print(f"Valor p: {coint_result[1]:.4f}")
    print(f"Valores críticos:")
    for i, value in enumerate(coint_result[2]):
        print(f"   {i+1}%: {value:.4f}")
    
    if coint_result[1] < 0.05:
        print(f"Conclusión: Las series están cointegradas (rechaza H0)")
    else:
        print(f"Conclusión: Las series no están cointegradas (no rechaza H0)")
    
    # Guardar resultados de cointegración
    with open(os.path.join(RESULTADOS_DIR, 'prueba_cointegracion.txt'), 'w') as f:
        f.write("PRUEBA DE COINTEGRACIÓN\n")
        f.write("======================\n\n")
        f.write(f"Estadístico: {coint_result[0]:.4f}\n")
        f.write(f"Valor p: {coint_result[1]:.4f}\n")
        f.write("Valores críticos:\n")
        for i, value in enumerate(coint_result[2]):
            f.write(f"   {i+1}%: {value:.4f}\n")
        
        if coint_result[1] < 0.05:
            f.write("\nConclusión: Las series están cointegradas (rechaza H0)")
        else:
            f.write("\nConclusión: Las series no están cointegradas (no rechaza H0)")
    
    print("Pruebas de estacionariedad completadas.")
    return df_conjunto

# Función para descomponer series temporales
def descomponer_series(df_conjunto):
    """
    Descompone las series temporales en componentes de tendencia, 
    estacionalidad y residuos
    """
    print("Descomponiendo series temporales...")
    
    # Descomposición de la serie de IVA
    descomp_iva = seasonal_decompose(df_conjunto['iva'], model='additive', period=12)
    
    # Visualizar descomposición de IVA
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Serie original
    descomp_iva.observed.plot(ax=axes[0], color='#006BA2')
    axes[0].set_ylabel('Observado')
    axes[0].set_title('Descomposición de Serie Temporal: IVA', fontweight='bold')
    
    # Tendencia
    descomp_iva.trend.plot(ax=axes[1], color='#A2C510')
    axes[1].set_ylabel('Tendencia')
    
    # Estacionalidad
    descomp_iva.seasonal.plot(ax=axes[2], color='#F4364C')
    axes[2].set_ylabel('Estacionalidad')
    
    # Residuos
    descomp_iva.resid.plot(ax=axes[3], color='#FF8C00')
    axes[3].set_ylabel('Residuos')
    
    # Configuración de eje X
    axes[3].set_xlabel('Año')
    axes[3].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'descomposicion_iva.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Descomposición de la serie de PIB
    descomp_pib = seasonal_decompose(df_conjunto['pib_usd'], model='additive', period=12)
    
    # Visualizar descomposición de PIB
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Serie original
    descomp_pib.observed.plot(ax=axes[0], color='#A2C510')
    axes[0].set_ylabel('Observado')
    axes[0].set_title('Descomposición de Serie Temporal: PIB', fontweight='bold')
    
    # Tendencia
    descomp_pib.trend.plot(ax=axes[1], color='#006BA2')
    axes[1].set_ylabel('Tendencia')
    
    # Estacionalidad
    descomp_pib.seasonal.plot(ax=axes[2], color='#F4364C')
    axes[2].set_ylabel('Estacionalidad')
    
    # Residuos
    descomp_pib.resid.plot(ax=axes[3], color='#FF8C00')
    axes[3].set_ylabel('Residuos')
    
    # Configuración de eje X
    axes[3].set_xlabel('Año')
    axes[3].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'descomposicion_pib.png'), dpi=100)
    plt.clf()
    plt.close(fig)
    
    # Guardar componentes
    componentes = pd.DataFrame({
        'iva_tendencia': descomp_iva.trend,
        'iva_estacional': descomp_iva.seasonal,
        'iva_residuo': descomp_iva.resid,
        'pib_tendencia': descomp_pib.trend,
        'pib_estacional': descomp_pib.seasonal,
        'pib_residuo': descomp_pib.resid
    })
    
    componentes.to_csv(os.path.join(RESULTADOS_DIR, 'componentes_series.csv'))
    
    print("Descomposición de series completada.")
    return descomp_iva, descomp_pib

# Función para modelar series temporales con ARIMA/SARIMA
def modelado_sarima_simple_iva(df_conjunto): # Renombrada de modelar_series
    """
    Modela la serie temporal del IVA utilizando un modelo SARIMA simple.
    Devuelve el resultado del modelo y un diccionario de métricas.
    """
    print("\n--- Modelado SARIMA simple para IVA ---")
    
    metricas_sarima = {}
    resultado_sarima_iva = None

    try:
        # Preparar datos para modelado
        fecha_corte = '2023-12-31' # O la última fecha disponible si es dinámica
        # Asegurarse de que df_conjunto.index sea DateTimeIndex
        if not isinstance(df_conjunto.index, pd.DatetimeIndex):
            df_conjunto.index = pd.to_datetime(df_conjunto.index)

        # Verificar si la fecha de corte existe en el índice
        if pd.to_datetime(fecha_corte) not in df_conjunto.index:
            # Si no existe, usar la última fecha disponible antes de la fecha de corte teórica
            # o ajustar la lógica según sea necesario. Aquí, por simplicidad, se podría tomar
            # un porcentaje de los datos o una fecha fija que se sepa que existe.
            # Para este ejemplo, si la fecha de corte no está, podríamos usar el 80% para entrenar.
            # Esto es una simplificación; una lógica más robusta sería necesaria en un caso real.
            split_point = int(len(df_conjunto) * 0.8)
            train_iva = df_conjunto['iva'].iloc[:split_point]
            test_iva = df_conjunto['iva'].iloc[split_point:]
            fecha_corte_real = train_iva.index[-1]
            print(f"Advertencia: Fecha de corte '{fecha_corte}' no encontrada. Usando '{fecha_corte_real}' como fecha de corte efectiva.")
        else:
            train_iva = df_conjunto.loc[:fecha_corte, 'iva']
            test_iva = df_conjunto.loc[pd.to_datetime(fecha_corte) + pd.DateOffset(days=1):, 'iva'] # Asegurar que test_iva comience después
            fecha_corte_real = pd.to_datetime(fecha_corte)


        train_iva = pd.to_numeric(train_iva, errors='coerce').dropna()
        test_iva = pd.to_numeric(test_iva, errors='coerce').dropna()
        
        if train_iva.empty:
            print("No hay datos de entrenamiento para el modelo SARIMA del IVA.")
            return None, {}

        # Visualizar ACF y PACF para IVA (opcional si ya se hizo en EDA)
        # ... (código de ACF/PACF existente) ...
        
        modelo_sarima_iva = SARIMAX(train_iva, 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    initialization='approximate_diffuse') # Añadido para consistencia
        
        resultado_sarima_iva = modelo_sarima_iva.fit(disp=False)
        
        print("\nResumen del modelo SARIMA para IVA:")
        print(resultado_sarima_iva.summary())
        with open(os.path.join(RESULTADOS_DIR, 'resumen_modelo_sarima_iva.txt'), 'w') as f:
            f.write(str(resultado_sarima_iva.summary()))
        
        # Pronóstico
        # Ajuste en muestra
        pred_train_iva = resultado_sarima_iva.get_prediction(start=train_iva.index[0], end=train_iva.index[-1], dynamic=False)
        pred_mean_train_iva = pred_train_iva.predicted_mean

        # Pronóstico fuera de muestra (test_iva)
        if not test_iva.empty:
            forecast_out_of_sample = resultado_sarima_iva.get_forecast(steps=len(test_iva))
            forecast_mean_iva = forecast_out_of_sample.predicted_mean
            forecast_ci_iva = forecast_out_of_sample.conf_int()
        else: # Si test_iva está vacío, pronosticar algunos pasos hacia el futuro
            print("Test set vacío, pronosticando 12 pasos futuros para SARIMA.")
            num_future_steps = 12
            future_forecast_index = pd.date_range(start=train_iva.index[-1] + pd.DateOffset(months=1), periods=num_future_steps, freq=train_iva.index.freqstr or 'MS')
            forecast_out_of_sample = resultado_sarima_iva.get_forecast(steps=num_future_steps)
            forecast_mean_iva = pd.Series(forecast_out_of_sample.predicted_mean.values, index=future_forecast_index)
            forecast_ci_iva = pd.DataFrame(forecast_out_of_sample.conf_int(), index=future_forecast_index, columns=['lower iva', 'upper iva'])


        # Visualizar resultados del modelo
        fig, ax = plt.figure(figsize=(14, 7)), plt.gca()
        ax.plot(df_conjunto.index, df_conjunto['iva'], color='#006BA2', label='Observado')
        ax.plot(pred_mean_train_iva.index, pred_mean_train_iva, color='#A2C510', linestyle='--', label='Ajustado (Entrenamiento)')
        if not test_iva.empty or num_future_steps > 0 :
            ax.plot(forecast_mean_iva.index, forecast_mean_iva, color='#F4364C', linestyle='--', label='Pronóstico')
            ax.fill_between(forecast_ci_iva.index, forecast_ci_iva.iloc[:, 0], forecast_ci_iva.iloc[:, 1], color='#F4364C', alpha=0.2)
        
        ax.axvline(x=fecha_corte_real, color='black', linestyle=':', label=f'Corte: {fecha_corte_real.strftime("%Y-%m-%d")}')
        ax.set_title('Modelo SARIMA para IVA: Ajuste y Pronóstico', fontweight='bold')
        # ... (resto de la configuración del gráfico como estaba) ...
        ax.set_xlabel('Año')
        ax.set_ylabel('IVA (Millones de pesos)')
        ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'modelo_sarima_iva.png'), dpi=100)
        plt.clf()
        plt.close(fig)
        
        # Evaluar modelo
        metricas_sarima['AIC'] = resultado_sarima_iva.aic
        metricas_sarima['BIC'] = resultado_sarima_iva.bic
        
        if not test_iva.empty and not forecast_mean_iva.empty:
            # Asegurar que los índices coinciden para la evaluación
            common_indices = test_iva.index.intersection(forecast_mean_iva.index)
            if not common_indices.empty:
                test_aligned = test_iva.loc[common_indices]
                forecast_aligned = forecast_mean_iva.loc[common_indices]
                
                if not test_aligned.empty and not forecast_aligned.empty:
                    metricas_sarima['RMSE'] = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
                    metricas_sarima['MAE'] = mean_absolute_error(test_aligned, forecast_aligned)
                    # MAPE puede dar problemas si test_aligned tiene ceros.
                    # metricas_sarima['MAPE'] = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                else:
                    print("Alineación de test y forecast resultó en series vacías para SARIMA.")
                    metricas_sarima['RMSE'] = np.nan
                    metricas_sarima['MAE'] = np.nan
            else:
                print("No hay índices comunes entre test y forecast para SARIMA.")
                metricas_sarima['RMSE'] = np.nan
                metricas_sarima['MAE'] = np.nan
        else: # Si no hay datos de prueba, las métricas de pronóstico no se calculan
            print("No hay datos de prueba para evaluar el pronóstico SARIMA, o el pronóstico está vacío.")
            metricas_sarima['RMSE'] = np.nan # O calcular RMSE en entrenamiento
            metricas_sarima['MAE'] = np.nan  # O calcular MAE en entrenamiento

        # Métricas en entrenamiento (siempre disponibles)
        rmse_train = np.sqrt(mean_squared_error(train_iva, pred_mean_train_iva))
        mae_train = mean_absolute_error(train_iva, pred_mean_train_iva)
        print(f"SARIMA - RMSE (entrenamiento): {rmse_train:.2f}")
        print(f"SARIMA - MAE (entrenamiento): {mae_train:.2f}")

        # Guardar métricas
        with open(os.path.join(RESULTADOS_DIR, 'metricas_modelo_sarima_iva.txt'), 'w') as f:
            f.write("EVALUACIÓN DEL MODELO SARIMA PARA IVA\n")
            f.write("=====================================\n\n")
            f.write(f"AIC: {metricas_sarima.get('AIC', 'N/A'):.2f}\n")
            f.write(f"BIC: {metricas_sarima.get('BIC', 'N/A'):.2f}\n")
            f.write(f"RMSE (entrenamiento): {rmse_train:.2f}\n")
            f.write(f"MAE (entrenamiento): {mae_train:.2f}\n")
            if 'RMSE' in metricas_sarima and not np.isnan(metricas_sarima['RMSE']):
                f.write(f"RMSE (prueba): {metricas_sarima['RMSE']:.2f}\n")
                f.write(f"MAE (prueba): {metricas_sarima['MAE']:.2f}\n")
            else:
                f.write("RMSE (prueba): N/A (no hay datos de prueba o error en cálculo)\n")
                f.write("MAE (prueba): N/A (no hay datos de prueba o error en cálculo)\n")
        
        # Análisis de residuos (código existente)
        # ... (código de diagnóstico de residuos existente) ...
        residuos_iva = resultado_sarima_iva.resid
        residuos_iva = residuos_iva.fillna(method='ffill').fillna(method='bfill')
        if not residuos_iva.empty:
            fig_diag = resultado_sarima_iva.plot_diagnostics(figsize=(15, 12))
            fig_diag.suptitle('Diagnóstico de Residuos del Modelo SARIMA para IVA', fontweight='bold', y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste para el supertítulo
            plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'diagnostico_residuos_sarima.png'), dpi=100)
            plt.clf()
            plt.close(fig_diag)
        else:
            print("Residuos vacíos, no se puede generar diagnóstico para SARIMA.")


    except Exception as e:
        print(f"Error en el modelado SARIMA: {e}")
        with open(os.path.join(RESULTADOS_DIR, 'error_modelado_sarima.txt'), 'w') as f:
            f.write(f"Error en el modelado SARIMA: {e}\n")
        return None, {} # Devuelve None y diccionario vacío en caso de error

    # La prueba de causalidad de Granger se mantiene como estaba, pero se podría mover
    # a una función separada o a main si depende de múltiples modelos.
    # ... (código de causalidad de Granger existente) ...

    print("Modelado SARIMA simple para IVA completado.")
    return resultado_sarima_iva, metricas_sarima

# Nueva función para modelado SARIMAX
def modelado_sarimax(iva_ts, pib_ts, nombre_modelo_base="SARIMAX IVA con PIB exógeno"):
    """
    Modela la serie temporal del IVA utilizando un modelo SARIMAX con el PIB como variable exógena.
    Devuelve el resultado del modelo y un diccionario de métricas.
    """
    print(f"\n--- Modelado {nombre_modelo_base} ---")
    
    metricas = {}
    results = None

    try:
        # Preparar datos para modelado
        fecha_corte = '2023-12-31' # O la última fecha disponible si es dinámica
        if not isinstance(iva_ts.index, pd.DatetimeIndex):
            iva_ts.index = pd.to_datetime(iva_ts.index)
        if not isinstance(pib_ts.index, pd.DatetimeIndex):
            pib_ts.index = pd.to_datetime(pib_ts.index)

        if pd.to_datetime(fecha_corte) not in iva_ts.index:
            split_point = int(len(iva_ts) * 0.8)
            train_iva = iva_ts.iloc[:split_point]
            test_iva = iva_ts.iloc[split_point:]
            train_pib = pib_ts.iloc[:split_point]
            test_pib = pib_ts.iloc[split_point:]
            fecha_corte_real = train_iva.index[-1]
            print(f"Advertencia: Fecha de corte '{fecha_corte}' no encontrada. Usando '{fecha_corte_real}' como fecha de corte efectiva.")
        else:
            train_iva = iva_ts.loc[:fecha_corte]
            test_iva = iva_ts.loc[pd.to_datetime(fecha_corte) + pd.DateOffset(days=1):]
            train_pib = pib_ts.loc[:fecha_corte]
            test_pib = pib_ts.loc[pd.to_datetime(fecha_corte) + pd.DateOffset(days=1):]
            fecha_corte_real = pd.to_datetime(fecha_corte)

        train_iva = pd.to_numeric(train_iva, errors='coerce').dropna()
        test_iva = pd.to_numeric(test_iva, errors='coerce').dropna()
        train_pib = pd.to_numeric(train_pib, errors='coerce').dropna()
        test_pib = pd.to_numeric(test_pib, errors='coerce').dropna()
        
        if train_iva.empty or train_pib.empty:
            print("No hay datos de entrenamiento suficientes para el modelo SARIMAX.")
            return None, {}

        # Modelado con auto_arima
        modelo_sarimax = pm.auto_arima(train_iva, exogenous=train_pib, seasonal=True, m=12,
                                       stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
        
        order_opt = modelo_sarimax.order
        seasonal_order_opt = modelo_sarimax.seasonal_order
        print(f"Órdenes óptimas para {nombre_modelo_base}: {order_opt} {seasonal_order_opt}")
        
        # Ajustar modelo SARIMAX con órdenes óptimas
        modelo_sarimax_opt = SARIMAX(train_iva, exog=train_pib, 
                                     order=order_opt, 
                                     seasonal_order=seasonal_order_opt,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False,
                                     initialization='approximate_diffuse')
        
        results = modelo_sarimax_opt.fit(disp=False)
        
        print(f"\nResumen del modelo {nombre_modelo_base}:")
        print(results.summary())
        with open(os.path.join(RESULTADOS_DIR, f'resumen_modelo_{nombre_modelo_base.lower().replace(" ", "_")}.txt'), 'w') as f:
            f.write(str(results.summary()))
        
        # Pronóstico
        pred_train = results.get_prediction(start=train_iva.index[0], end=train_iva.index[-1], exog=train_pib, dynamic=False)
        pred_mean_train = pred_train.predicted_mean

        if not test_iva.empty and not test_pib.empty:
            forecast_out_of_sample = results.get_forecast(steps=len(test_iva), exog=test_pib)
            forecast_mean = forecast_out_of_sample.predicted_mean
            forecast_ci = forecast_out_of_sample.conf_int()
        else:
            print(f"Test set vacío, pronosticando 12 pasos futuros para {nombre_modelo_base}.")
            num_future_steps = 12
            future_forecast_index = pd.date_range(start=train_iva.index[-1] + pd.DateOffset(months=1), periods=num_future_steps, freq=train_iva.index.freqstr or 'MS')
            future_pib = np.tile(train_pib.values[-12:], num_future_steps // 12 + 1)[:num_future_steps]
            forecast_out_of_sample = results.get_forecast(steps=num_future_steps, exog=future_pib.reshape(-1, 1))
            forecast_mean = pd.Series(forecast_out_of_sample.predicted_mean.values, index=future_forecast_index)
            forecast_ci = pd.DataFrame(forecast_out_of_sample.conf_int(), index=future_forecast_index, columns=['lower iva', 'upper iva'])

        # Visualizar resultados del modelo
        fig, ax = plt.figure(figsize=(14, 7)), plt.gca()
        ax.plot(iva_ts.index, iva_ts, color='#006BA2', label='Observado')
        ax.plot(pred_mean_train.index, pred_mean_train, color='#A2C510', linestyle='--', label='Ajustado (Entrenamiento)')
        if not test_iva.empty or num_future_steps > 0:
            ax.plot(forecast_mean.index, forecast_mean, color='#F4364C', linestyle='--', label='Pronóstico')
            ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='#F4364C', alpha=0.2)
        
        ax.axvline(x=fecha_corte_real, color='black', linestyle=':', label=f'Corte: {fecha_corte_real.strftime("%Y-%m-%d")}')
        ax.set_title(f'Modelo {nombre_modelo_base}: Ajuste y Pronóstico', fontweight='bold')
        ax.set_xlabel('Año')
        ax.set_ylabel('IVA (Millones de pesos)')
        ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, f'modelo_{nombre_modelo_base.lower().replace(" ", "_")}.png'), dpi=100)
        plt.clf()
        plt.close(fig)
        
        # Evaluar modelo
        metricas['AIC'] = results.aic
        metricas['BIC'] = results.bic
        
        if not test_iva.empty and not forecast_mean.empty:
            common_indices = test_iva.index.intersection(forecast_mean.index)
            if not common_indices.empty:
                test_aligned = test_iva.loc[common_indices]
                forecast_aligned = forecast_mean.loc[common_indices]
                
                if not test_aligned.empty and not forecast_aligned.empty:
                    metricas['RMSE'] = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
                    metricas['MAE'] = mean_absolute_error(test_aligned, forecast_aligned)
                else:
                    print(f"Alineación de test y forecast resultó en series vacías para {nombre_modelo_base}.")
                    metricas['RMSE'] = np.nan
                    metricas['MAE'] = np.nan
            else:
                print(f"No hay índices comunes entre test y forecast para {nombre_modelo_base}.")
                metricas['RMSE'] = np.nan
                metricas['MAE'] = np.nan
        else:
            print(f"No hay datos de prueba para evaluar el pronóstico {nombre_modelo_base}, o el pronóstico está vacío.")
            metricas['RMSE'] = np.nan
            metricas['MAE'] = np.nan

        rmse_train = np.sqrt(mean_squared_error(train_iva, pred_mean_train))
        mae_train = mean_absolute_error(train_iva, pred_mean_train)
        print(f"{nombre_modelo_base} - RMSE (entrenamiento): {rmse_train:.2f}")
        print(f"{nombre_modelo_base} - MAE (entrenamiento): {mae_train:.2f}")

        with open(os.path.join(RESULTADOS_DIR, f'metricas_modelo_{nombre_modelo_base.lower().replace(" ", "_")}.txt'), 'w') as f:
            f.write(f"EVALUACIÓN DEL MODELO {nombre_modelo_base.upper()}\n")
            f.write("=====================================\n\n")
            f.write(f"AIC: {metricas.get('AIC', 'N/A'):.2f}\n")
            f.write(f"BIC: {metricas.get('BIC', 'N/A'):.2f}\n")
            f.write(f"RMSE (entrenamiento): {rmse_train:.2f}\n")
            f.write(f"MAE (entrenamiento): {mae_train:.2f}\n")
            if 'RMSE' in metricas and not np.isnan(metricas['RMSE']):
                f.write(f"RMSE (prueba): {metricas['RMSE']:.2f}\n")
                f.write(f"MAE (prueba): {metricas['MAE']:.2f}\n")
            else:
                f.write("RMSE (prueba): N/A (no hay datos de prueba o error en cálculo)\n")
                f.write("MAE (prueba): N/A (no hay datos de prueba o error en cálculo)\n")

        residuos = results.resid
        residuos = residuos.fillna(method='ffill').fillna(method='bfill')
        if not residuos.empty:
            fig_diag = results.plot_diagnostics(figsize=(15, 12))
            fig_diag.suptitle(f'Diagnóstico de Residuos del Modelo {nombre_modelo_base}', fontweight='bold', y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(VISUALIZACIONES_DIR, f'diagnostico_residuos_{nombre_modelo_base.lower().replace(" ", "_")}.png'), dpi=100)
            plt.clf()
            plt.close(fig_diag)
        else:
            print(f"Residuos vacíos, no se puede generar diagnóstico para {nombre_modelo_base}.")

        metricas = {
            'AIC': results.aic,
            'BIC': results.bic,
            'MAE': mae_train,
            'RMSE': rmse_train,
            'Order': str(order_opt),  # Convertir a string para consistencia en CSV
            'SeasonalOrder': str(seasonal_order_opt) # Convertir a string
        }

    except Exception as e:
        print(f"Error en el modelado {nombre_modelo_base}: {e}")
        with open(os.path.join(RESULTADOS_DIR, f'error_modelado_{nombre_modelo_base.lower().replace(" ", "_")}.txt'), 'w') as f:
            f.write(f"Error en el modelado {nombre_modelo_base}: {e}\n")
        return None, {}

    print(f"Modelado {nombre_modelo_base} completado.")
    return results, metricas

# Nueva función para generar comparación de modelos
def generar_comparacion_modelos(metricas_modelos):
    """
    Genera un archivo CSV y un gráfico comparando las métricas de los modelos.
    metricas_modelos: lista de diccionarios, cada uno con 'nombre' y métricas.
                      Ej: [{'nombre': 'SARIMA', 'AIC': ..., 'RMSE': ...}, ...]
    """
    print("\n--- Generando Comparación de Modelos ---")
    if not metricas_modelos:
        print("No hay métricas de modelos para comparar.")
        return

    # Crear DataFrame con las métricas
    df_comparacion = pd.DataFrame(metricas_modelos)
    
    # Seleccionar métricas relevantes para el CSV y asegurar que las columnas existan
    columnas_csv = ['nombre', 'AIC', 'BIC', 'RMSE', 'MAE', 'Order', 'SeasonalOrder']
    columnas_presentes = [col for col in columnas_csv if col in df_comparacion.columns]
    df_comparacion_csv = df_comparacion[columnas_presentes]

    # Guardar en CSV
    ruta_csv = os.path.join(RESULTADOS_DIR, 'comparacion_modelos.csv')
    try:
        df_comparacion_csv.to_csv(ruta_csv, index=False, float_format='%.2f')
        print(f"Archivo de comparación de modelos guardado en: {ruta_csv}")
    except Exception as e:
        print(f"Error al guardar comparacion_modelos.csv: {e}")

    # Generar gráfico de comparación (RMSE y MAE)
    # Filtrar modelos que tengan RMSE y MAE válidos
    df_plot = df_comparacion[df_comparacion['RMSE'].notna() & df_comparacion['MAE'].notna()].copy()

    if df_plot.empty:
        print("No hay suficientes datos válidos de RMSE/MAE para generar el gráfico de comparación.")
        return

    # Usar nombres de modelo como índice para el gráfico
    df_plot.set_index('nombre', inplace=True)
    
    # Seleccionar solo RMSE y MAE para el gráfico
    df_plot_metrics = df_plot[['RMSE', 'MAE']]

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot_metrics.plot(kind='bar', ax=ax, colormap='viridis', alpha=0.75) # Usar un colormap
        
        ax.set_title('Comparación de Modelos: RMSE y MAE', fontweight='bold', fontsize=14)
        ax.set_ylabel('Valor de la Métrica', fontsize=12)
        ax.set_xlabel('Modelo', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Métricas')
        
        # Añadir valores en las barras
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points', fontsize=8)

        plt.tight_layout()
        ruta_png = os.path.join(VISUALIZACIONES_DIR, 'comparacion_modelos.png')
        plt.savefig(ruta_png, dpi=100)
        plt.clf()
        plt.close(fig)
        print(f"Gráfico de comparación de modelos guardado en: {ruta_png}")
    except Exception as e:
        print(f"Error al generar comparacion_modelos.png: {e}")

# Función principal
def main():
    """Función principal para el análisis de series temporales"""
    print("Iniciando análisis de series temporales: PIB e IVA en Colombia (2000-2024)...")
    
    df_conjunto = cargar_datos_alineados()
    if df_conjunto.empty:
        print("No se pudieron cargar los datos. Abortando análisis.")
        return

    realizar_eda(df_conjunto.copy())
    pruebas_estacionariedad(df_conjunto.copy())
    descomponer_series(df_conjunto.copy())
    
    iva_ts = df_conjunto['iva'].copy()
    pib_ts = df_conjunto['pib_usd'].copy()

    lista_metricas_comparacion = []

    # Modelado SARIMA simple
    print("\nLlamando a modelado_sarima_simple_iva...")
    # Pasar df_conjunto completo para que la función maneje el split train/test
    resultado_sarima_simple, metricas_sarima = modelado_sarima_simple_iva(df_conjunto.copy()) 
    if resultado_sarima_simple and metricas_sarima:
        print("\n--- Resultados del Modelo SARIMA Simple ---")
        # print(resultado_sarima_simple.summary()) # Ya se imprime y guarda dentro de la función
        print("\nMétricas del modelo SARIMA simple:")
        for k, v in metricas_sarima.items():
            print(f"  {k}: {v}")
        metricas_sarima_comp = {'nombre': 'SARIMA (IVA)'}
        metricas_sarima_comp.update(metricas_sarima)
        # Extraer órdenes si están disponibles en el resultado del modelo (no en auto_arima)
        # Para SARIMAX manual, los órdenes son fijos.
        metricas_sarima_comp['Order'] = str(resultado_sarima_simple.model.order) if hasattr(resultado_sarima_simple, 'model') else 'N/A'
        metricas_sarima_comp['SeasonalOrder'] = str(resultado_sarima_simple.model.seasonal_order) if hasattr(resultado_sarima_simple, 'model') else 'N/A'
        lista_metricas_comparacion.append(metricas_sarima_comp)
    else:
        print("El modelado SARIMA simple no pudo completarse o no devolvió resultados/métricas.")

    # Modelado SARIMAX
    if not iva_ts.empty and not pib_ts.empty:
        print("\nLlamando a modelado_sarimax...")
        # Para SARIMAX, usualmente se entrena con todos los datos disponibles de IVA y PIB alineados
        # y luego se pronostica. La función modelado_sarimax ya maneja esto.
        modelo_sarimax_results, metricas_sarimax = modelado_sarimax(iva_ts, pib_ts) # Usa iva_ts y pib_ts completos
        if modelo_sarimax_results and metricas_sarimax:
            print("\n--- Resultados del Modelo SARIMAX ---")
            # print(modelo_sarimax_results.summary()) # Ya se imprime y guarda dentro de la función
            print("\nMétricas del modelo SARIMAX:")
            for k, v in metricas_sarimax.items(): # metricas_sarimax ya tiene Order y SeasonalOrder de auto_arima
                print(f"  {k}: {v}")
            metricas_sarimax_comp = {'nombre': 'SARIMAX (IVA ~ PIB)'}
            metricas_sarimax_comp.update(metricas_sarimax)
            lista_metricas_comparacion.append(metricas_sarimax_comp)
        else:
            print("El modelado SARIMAX no pudo completarse o no devolvió resultados/métricas.")
    else:
        print("Las series de IVA o PIB están vacías antes de llamar a modelado_sarimax. Saltando modelado SARIMAX.")

    # Generar comparación de modelos si hay métricas disponibles
    if lista_metricas_comparacion:
        generar_comparacion_modelos(lista_metricas_comparacion)
    else:
        print("No hay métricas de ningún modelo para comparar.")
    
    print("\nAnálisis de series temporales completado con éxito.")
    print(f"Resultados guardados en las carpetas '{RESULTADOS_DIR}' y '{VISUALIZACIONES_DIR}'.")

if __name__ == "__main__":
    main()
