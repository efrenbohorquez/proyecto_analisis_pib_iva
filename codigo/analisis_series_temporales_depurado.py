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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_temporales_iva_pib.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'dispersion_iva_pib.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'variacion_porcentual_anual.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'estacionalidad_mensual_iva.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_diferenciadas.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'descomposicion_iva.png'), dpi=300)
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
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'descomposicion_pib.png'), dpi=300)
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
    """
    print("\n--- Modelado SARIMA simple para IVA ---")
    
    # Preparar datos para modelado
    fecha_corte = '2023-12-31'
    train_iva = df_conjunto.loc[:fecha_corte, 'iva']
    test_iva = df_conjunto.loc[fecha_corte:, 'iva']
    
    train_iva = pd.to_numeric(train_iva, errors='coerce').dropna()
    
    if train_iva.empty:
        print("No hay datos de entrenamiento para el modelo SARIMA del IVA.")
        return None

    # Visualizar ACF y PACF para IVA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train_iva, ax=ax1, lags=36)
    ax1.set_title('Función de Autocorrelación (ACF) - IVA', fontweight='bold')
    
    plot_pacf(train_iva, ax=ax2, lags=36)
    ax2.set_title('Función de Autocorrelación Parcial (PACF) - IVA', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'acf_pacf_iva.png'), dpi=300)
    plt.close(fig)
    
    # Modelado SARIMA para IVA
    # Basado en análisis ACF/PACF y conocimiento del dominio
    # Parámetros: (p,d,q)x(P,D,Q,s)
    try:
        modelo_sarima_iva = SARIMAX(train_iva, 
                                order=(1, 1, 1), 
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        
        resultado_sarima_iva = modelo_sarima_iva.fit(disp=False)
        
        # Resumen del modelo
        print("\nResumen del modelo SARIMA para IVA:")
        print(resultado_sarima_iva.summary())
        
        # Guardar resumen del modelo
        with open(os.path.join(RESULTADOS_DIR, 'resumen_modelo_sarima_iva.txt'), 'w') as f:
            f.write(str(resultado_sarima_iva.summary()))
        
        # Pronóstico dentro de la muestra
        pred_iva = resultado_sarima_iva.predict(start=train_iva.index[0], end=df_conjunto.index[-1])
        
        # Pronóstico fuera de la muestra (2024)
        forecast_iva = resultado_sarima_iva.get_forecast(steps=len(test_iva))
        forecast_mean_iva = forecast_iva.predicted_mean
        forecast_ci_iva = forecast_iva.conf_int()
        
        # Verificar y limpiar datos para visualización
        # Asegurar que no hay valores NaN o infinitos
        pred_iva = pred_iva.fillna(method='ffill').fillna(method='bfill')
        forecast_mean_iva = forecast_mean_iva.fillna(method='ffill').fillna(method='bfill')
        forecast_ci_iva = forecast_ci_iva.fillna(method='ffill').fillna(method='bfill')
        
        # Visualizar resultados del modelo
        fig, ax = plt.figure(figsize=(14, 7)), plt.gca()
        
        # Datos originales
        ax.plot(df_conjunto.index, df_conjunto['iva'], 
                color='#006BA2', label='Observado')
        
        # Valores ajustados
        ax.plot(pred_iva.index, pred_iva, 
                color='#A2C510', linestyle='--', label='Ajustado')
        
        # Pronóstico
        ax.plot(forecast_mean_iva.index, forecast_mean_iva, 
                color='#F4364C', linestyle='--', label='Pronóstico')
        
        # Intervalo de confianza
        ax.fill_between(forecast_ci_iva.index,
                       forecast_ci_iva.iloc[:, 0],
                       forecast_ci_iva.iloc[:, 1],
                       color='#F4364C', alpha=0.2)
        
        # Línea vertical para separar entrenamiento y prueba
        ax.axvline(x=pd.to_datetime(fecha_corte), color='black', linestyle=':')
        
        # Configuración del gráfico
        ax.set_title('Modelo SARIMA para IVA: Valores Ajustados y Pronóstico', fontweight='bold')
        ax.set_xlabel('Año')
        ax.set_ylabel('IVA (Millones de pesos)')
        ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Configuración de eje X
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'modelo_sarima_iva.png'), dpi=300)
        plt.close(fig)
        
        # Evaluar modelo
        # Asegurar que los índices coinciden exactamente
        test_indices = test_iva.index
        forecast_indices = forecast_mean_iva.index
        
        # Encontrar índices comunes
        common_indices = test_indices.intersection(forecast_indices)
        
        if len(common_indices) > 0:
            test_aligned = test_iva.loc[common_indices]
            forecast_aligned = forecast_mean_iva.loc[common_indices]
            
            # Calcular métricas
            rmse_iva = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
            mae_iva = mean_absolute_error(test_aligned, forecast_aligned)
            mape_iva = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
            
            print("\nEvaluación del modelo SARIMA para IVA:")
            print(f"RMSE: {rmse_iva:.2f}")
            print(f"MAE: {mae_iva:.2f}")
            print(f"MAPE: {mape_iva:.2f}%")
            
            # Guardar métricas
            with open(os.path.join(RESULTADOS_DIR, 'metricas_modelo_sarima_iva.txt'), 'w') as f:
                f.write("EVALUACIÓN DEL MODELO SARIMA PARA IVA\n")
                f.write("=====================================\n\n")
                f.write(f"RMSE: {rmse_iva:.2f}\n")
                f.write(f"MAE: {mae_iva:.2f}\n")
                f.write(f"MAPE: {mape_iva:.2f}%\n")
        else:
            print("No hay suficientes datos para evaluar el modelo")
            with open(os.path.join(RESULTADOS_DIR, 'metricas_modelo_sarima_iva.txt'), 'w') as f:
                f.write("EVALUACIÓN DEL MODELO SARIMA PARA IVA\n")
                f.write("=====================================\n\n")
                f.write("No hay suficientes datos para evaluar el modelo\n")
        
        # Análisis de residuos
        residuos_iva = resultado_sarima_iva.resid
        
        # Asegurar que no hay valores NaN o infinitos
        residuos_iva = residuos_iva.fillna(method='ffill').fillna(method='bfill')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico de residuos
        axes[0, 0].plot(residuos_iva, color='#006BA2')
        axes[0, 0].set_title('Residuos del Modelo SARIMA', fontweight='bold')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Residuo')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[0, 1].hist(residuos_iva, bins=20, color='#006BA2', alpha=0.7)
        axes[0, 1].set_title('Histograma de Residuos', fontweight='bold')
        axes[0, 1].set_xlabel('Residuo')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF de residuos
        plot_acf(residuos_iva, ax=axes[1, 0], lags=36)
        axes[1, 0].set_title('ACF de Residuos', fontweight='bold')
        
        # QQ plot
        sm.qqplot(residuos_iva, line='45', ax=axes[1, 1])
        axes[1, 1].set_title('QQ Plot de Residuos', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'diagnostico_residuos_sarima.png'), dpi=300)
        plt.close(fig)
        
        # Diagnóstico de residuos
        fig = resultado_sarima_iva.plot_diagnostics(figsize=(15, 12))
        fig.suptitle('Diagnóstico de Residuos del Modelo SARIMA para IVA', fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'diagnostico_residuos_sarima.png'), dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Error en el modelado SARIMA: {e}")
        # Guardar error en archivo
        with open(os.path.join(RESULTADOS_DIR, 'error_modelado.txt'), 'w') as f:
            f.write(f"Error en el modelado SARIMA: {e}\n")
            f.write("No se pudo completar el modelado SARIMA ni las pruebas dependientes.")
        # Continuar con el resto del script si es posible, o manejar el error
        # Por ejemplo, se podría asignar un valor por defecto a las variables que no se pudieron calcular
        resultado_sarima_iva = None # o algún otro valor que indique fallo
        # ... otras variables que dependan de esto ...

    # Prueba de causalidad de Granger
    # Solo si el modelo SARIMA se ejecutó correctamente
    if resultado_sarima_iva is not None:
        print("Realizando prueba de causalidad de Granger...")
        
        # Preparar datos para la prueba de Granger
        # Usar las series diferenciadas si las originales no son estacionarias
        # Asegurarse de que ambas series tengan la misma longitud y no contengan NaNs
        data_granger = df_conjunto[['iva_diff', 'pib_diff']].copy()
        data_granger = data_granger.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        if not data_granger.empty and len(data_granger) > 15: # Se necesita un mínimo de datos
            try:
                # Convertir a tipos de datos compatibles si es necesario
                data_granger['iva_diff'] = pd.to_numeric(data_granger['iva_diff'], errors='coerce')
                data_granger['pib_diff'] = pd.to_numeric(data_granger['pib_diff'], errors='coerce')
                data_granger.dropna(inplace=True)

                if len(data_granger) > 15: # Re-verificar después de la coerción y dropna
                    max_lags = min(12, len(data_granger) // 4) # Limitar lags para evitar errores
                    if max_lags > 0:
                        granger_results_pib_iva = grangercausalitytests(data_granger[['pib_diff', 'iva_diff']], 
                                                                        maxlag=max_lags, verbose=False)
                        granger_results_iva_pib = grangercausalitytests(data_granger[['iva_diff', 'pib_diff']], 
                                                                        maxlag=max_lags, verbose=False)
                        
                        # Guardar resultados de causalidad de Granger
                        with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
                            f.write("PRUEBA DE CAUSALIDAD DE GRANGER\n")
                            f.write("==============================\n\n")
                            f.write("Hipótesis: PIB no causa IVA\n")
                            for lag in granger_results_pib_iva:
                                test_stat = granger_results_pib_iva[lag][0]['ssr_ftest'][0]
                                p_value = granger_results_pib_iva[lag][0]['ssr_ftest'][1]
                                f.write(f"Lag {lag}: Estadístico F = {test_stat:.4f}, Valor p = {p_value:.4f}\n")
                            
                            f.write("\nHipótesis: IVA no causa PIB\n")
                            for lag in granger_results_iva_pib:
                                test_stat = granger_results_iva_pib[lag][0]['ssr_ftest'][0]
                                p_value = granger_results_iva_pib[lag][0]['ssr_ftest'][1]
                                f.write(f"Lag {lag}: Estadístico F = {test_stat:.4f}, Valor p = {p_value:.4f}\n")
                        print("Prueba de causalidad de Granger completada.")
                    else:
                        print("No hay suficientes datos para la prueba de Granger después de la limpieza.")
                        with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
                            f.write("No hay suficientes datos para la prueba de Granger después de la limpieza.")
                else:
                    print("No hay suficientes datos para la prueba de Granger después de la conversión y limpieza.")
                    with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
                        f.write("No hay suficientes datos para la prueba de Granger después de la conversión y limpieza.")

            except Exception as e_granger:
                print(f"Error en la prueba de causalidad de Granger: {e_granger}")
                with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
                    f.write(f"Error en la prueba de causalidad de Granger: {e_granger}\n")
        else:
            print("No hay suficientes datos o datos vacíos para la prueba de Granger.")
            with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
                f.write("No hay suficientes datos o datos vacíos para la prueba de Granger.\n")
    else:
        print("Modelado SARIMA falló, omitiendo prueba de causalidad de Granger.")
        with open(os.path.join(RESULTADOS_DIR, 'causalidad_granger.txt'), 'w') as f:
            f.write("Modelado SARIMA falló, omitiendo prueba de causalidad de Granger.\n")

    print("Modelado SARIMA simple para IVA completado.")
    return resultado_sarima_iva # Asegúrate que esta variable se defina correctamente dentro de la función original

# Nueva función para modelado SARIMAX
def modelado_sarimax(iva_ts, pib_ts, nombre_modelo_base="SARIMAX IVA con PIB exógeno"):
    """
    Realiza el modelado SARIMAX para la serie del IVA utilizando el PIB como variable exógena.
    Encuentra el mejor modelo usando auto_arima, lo ajusta y guarda resultados y gráficos.
    """
    print(f"\n--- Modelado {nombre_modelo_base} ---")
    
    if iva_ts.empty or pib_ts.empty:
        print("Error: Las series de IVA o PIB están vacías. No se puede continuar con el modelado SARIMAX.")
        return None, {}

    # Asegurar que pib_ts (exógena) esté alineada con iva_ts y no tenga NaNs donde iva_ts no los tiene.
    pib_ts_aligned = pib_ts[iva_ts.index].dropna()
    iva_ts_aligned = iva_ts[pib_ts_aligned.index].dropna() # Re-alinear iva_ts con los índices válidos de pib_ts
    
    if iva_ts_aligned.empty or pib_ts_aligned.empty:
        print("Error: Series vacías después de alineación y dropna para SARIMAX. No se puede modelar.")
        return None, {}
        
    exog_pib_reshaped = pib_ts_aligned.values.reshape(-1, 1)

    print("Buscando el mejor modelo SARIMAX con auto_arima...")
    try:
        modelo_auto = pm.auto_arima(iva_ts_aligned, 
                                    exogenous=exog_pib_reshaped,
                                    start_p=1, start_q=1,
                                    test='adf',
                                    max_p=3, max_q=3,
                                    m=12, # Frecuencia mensual
                                    start_P=0, seasonal=True,
                                    D=None, # auto_arima determina D
                                    trace=True,
                                    error_action='ignore',  
                                    suppress_warnings=True, 
                                    stepwise=True)

        print(f"Mejor modelo SARIMAX encontrado: {modelo_auto.order}, {modelo_auto.seasonal_order}")
        order_opt = modelo_auto.order
        seasonal_order_opt = modelo_auto.seasonal_order
        
        model = SARIMAX(iva_ts_aligned,
                        exog=exog_pib_reshaped,
                        order=order_opt,
                        seasonal_order=seasonal_order_opt,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        initialization='approximate_diffuse')
        results = model.fit(disp=False)

    except Exception as e:
        print(f"Error durante auto_arima o ajuste del modelo SARIMAX: {e}")
        print("Intentando con parámetros SARIMAX por defecto (1,1,1)(1,1,1,12).")
        try:
            order_opt = (1,1,1)
            seasonal_order_opt = (1,1,1,12)
            model = SARIMAX(iva_ts_aligned,
                            exog=exog_pib_reshaped,
                            order=order_opt,
                            seasonal_order=seasonal_order_opt,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            initialization='approximate_diffuse')
            results = model.fit(disp=False)
        except Exception as e_fallback:
            print(f"Error con el modelo SARIMAX de fallback: {e_fallback}")
            return None, {}

    print(results.summary())
    with open(os.path.join(RESULTADOS_DIR, 'resumen_modelo_sarimax_iva.txt'), 'w') as f: # Nombre corregido
        f.write(results.summary().as_text())
    print(f"Resumen del modelo SARIMAX guardado en {os.path.join(RESULTADOS_DIR, 'resumen_modelo_sarimax_iva.txt')}")

    try:
        fig_diag = results.plot_diagnostics(figsize=(15, 12))
        fig_diag.suptitle(f'Diagnóstico de Residuos del Modelo {nombre_modelo_base}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'diagnostico_residuos_sarimax.png')) # Nombre corregido
        plt.close(fig_diag)
        print(f"Diagnóstico de residuos SARIMAX guardado en {os.path.join(VISUALIZACIONES_DIR, 'diagnostico_residuos_sarimax.png')}")
    except Exception as e:
        print(f"Error al generar diagnóstico de residuos SARIMAX: {e}")

    pred_insample = results.get_prediction(start=iva_ts_aligned.index[0], 
                                           end=iva_ts_aligned.index[-1], 
                                           exog=exog_pib_reshaped, 
                                           dynamic=False)
    pred_mean_insample = pred_insample.predicted_mean
    
    mae = mean_absolute_error(iva_ts_aligned, pred_mean_insample)
    rmse = np.sqrt(mean_squared_error(iva_ts_aligned, pred_mean_insample))
    aic = results.aic
    bic = results.bic
    
    print(f"AIC (SARIMAX): {aic}")
    print(f"BIC (SARIMAX): {bic}")
    print(f"MAE (SARIMAX en muestra): {mae}")
    print(f"RMSE (SARIMAX en muestra): {rmse}")

    with open(os.path.join(RESULTADOS_DIR, 'metricas_modelo_sarimax_iva.txt'), 'w') as f: # Nombre corregido
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
        f.write(f"MAE (en muestra): {mae}\n")
        f.write(f"RMSE (en muestra): {rmse}\n")
        f.write(f"Order: {order_opt}\n")
        f.write(f"Seasonal Order: {seasonal_order_opt}\n")
    print(f"Métricas del modelo SARIMAX guardadas en {os.path.join(RESULTADOS_DIR, 'metricas_modelo_sarimax_iva.txt')}")

    n_forecast = 24 # Aumentado para mejor visualización
    
    # Crear exógenos para el período de pronóstico (ej. usando el último valor conocido de PIB o una proyección)
    # Aquí, como ejemplo simple, repetimos el último valor. En un caso real, se usaría una predicción del PIB.
    last_pib_value = pib_ts_aligned.iloc[-1] if not pib_ts_aligned.empty else 0
    future_exog_array = np.array([last_pib_value] * n_forecast).reshape(-1, 1)
    
    last_date_iva = iva_ts_aligned.index[-1] if not iva_ts_aligned.empty else pd.Timestamp.now()
    forecast_index = pd.date_range(start=last_date_iva + pd.DateOffset(months=1), 
                                   periods=n_forecast, 
                                   freq=iva_ts_aligned.index.freqstr or 'MS')

    pred_uc = results.get_forecast(steps=n_forecast, exog=future_exog_array)
    pred_ci = pred_uc.conf_int()

    plt.figure(figsize=(14, 7))
    plt.plot(iva_ts_aligned, label='Observado (IVA)', color='#006BA2')
    plt.plot(pred_mean_insample, label='Ajuste SARIMAX en muestra', color='#A2C510', linestyle='--')
    
    pred_uc_mean_series = pd.Series(pred_uc.predicted_mean.values, index=forecast_index)
    plt.plot(pred_uc_mean_series, label=f'Predicción SARIMAX ({n_forecast} meses)', color='#F4364C', linestyle='--')
    
    pred_ci.index = forecast_index
    plt.fill_between(pred_ci.index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1], color='#F4364C', alpha=.15, label='Intervalo de Confianza')
    
    plt.title(f'Modelo {nombre_modelo_base}: Observado vs. Ajustado vs. Predicción', fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Recaudo IVA (Millones COP)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(formato_millones)) # Usar formateador
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'modelo_sarimax_iva.png')) # Nombre corregido
    plt.close()
    print(f"Gráfico del modelo SARIMAX guardado en {os.path.join(VISUALIZACIONES_DIR, 'modelo_sarimax_iva.png')}")
    
    metricas = {'AIC': aic, 'BIC': bic, 'MAE': mae, 'RMSE': rmse, 
                'Order': order_opt, 'SeasonalOrder': seasonal_order_opt}
    return results, metricas

# Función principal
def main():
    """Función principal para el análisis de series temporales"""
    print("Iniciando análisis de series temporales: PIB e IVA en Colombia (2000-2024)...")
    
    # Cargar datos alineados
    df_conjunto = cargar_datos_alineados() # df_conjunto tiene 'iva' y 'pib_usd'
    
    # Realizar análisis exploratorio
    df_conjunto_eda = realizar_eda(df_conjunto.copy()) # Pasar copia para no modificar original innecesariamente
    
    # Realizar pruebas de estacionariedad
    df_conjunto_est = pruebas_estacionariedad(df_conjunto.copy())
    
    # Descomponer series temporales
    descomp_iva, descomp_pib = descomponer_series(df_conjunto.copy())
    
    # Modelar series temporales con SARIMAX
    # Extraer las series individuales para SARIMAX
    iva_ts = df_conjunto['iva'].copy()
    pib_ts = df_conjunto['pib_usd'].copy() # Usar 'pib_usd' como exógena

    if not iva_ts.empty and not pib_ts.empty:
        print("\nLlamando a modelado_sarimax...")
        modelo_sarimax_results, metricas_sarimax = modelado_sarimax(iva_ts, pib_ts)
        if modelo_sarimax_results:
            print("\n--- Resultados del Modelo SARIMAX ---")
            print(modelo_sarimax_results.summary())
            print("\nMétricas del modelo SARIMAX:")
            for k, v in metricas_sarimax.items():
                print(f"  {k}: {v}")
        else:
            print("El modelado SARIMAX no pudo completarse o retornó None.")
    else:
        print("Las series de IVA o PIB están vacías antes de llamar a modelado_sarimax. Saltando modelado.")

    # Opcional: Llamar al modelado SARIMA simple si se desea conservar
    # print("\nLlamando a modelado_sarima_simple_iva...")
    # resultado_sarima_simple = modelado_sarima_simple_iva(df_conjunto.copy())
    # if resultado_sarima_simple:
    #     print("\n--- Resultados del Modelo SARIMA Simple ---")
    #     print(resultado_sarima_simple.summary())
    
    print("\nAnálisis de series temporales (con enfoque en SARIMAX) completado con éxito.")
    print(f"Resultados guardados en las carpetas '{RESULTADOS_DIR}' y '{VISUALIZACIONES_DIR}'.")

if __name__ == "__main__":
    main()
