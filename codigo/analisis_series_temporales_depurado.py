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
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

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

# Crear directorios si no existen
os.makedirs('../visualizaciones', exist_ok=True)
os.makedirs('../resultados', exist_ok=True)

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
    df_conjunto = pd.read_csv('../datos/iva_pib_alineado.csv')
    
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
    with open('../resultados/estadisticas_descriptivas.txt', 'w') as f:
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
    plt.savefig('../visualizaciones/series_temporales_iva_pib.png', dpi=300)
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
    plt.savefig('../visualizaciones/dispersion_iva_pib.png', dpi=300)
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
    plt.savefig('../visualizaciones/variacion_porcentual_anual.png', dpi=300)
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
    plt.savefig('../visualizaciones/estacionalidad_mensual_iva.png', dpi=300)
    plt.close(fig)
    
    # Calcular correlación
    correlacion = df_conjunto['iva'].corr(df_conjunto['pib_usd'])
    print(f"Correlación entre IVA y PIB: {correlacion:.4f}")
    
    # Guardar correlación
    with open('../resultados/correlacion_iva_pib.txt', 'w') as f:
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
    plt.savefig('../visualizaciones/series_diferenciadas.png', dpi=300)
    plt.close(fig)
    
    # Pruebas para series diferenciadas
    print("\n" + "="*50)
    print("PRUEBAS DE ESTACIONARIEDAD PARA SERIES DIFERENCIADAS")
    print("="*50)
    adf_iva_diff = test_adf(df_diff['iva_diff'], 'IVA diferenciado')
    adf_pib_diff = test_adf(df_diff['pib_diff'], 'PIB diferenciado')
    
    # Guardar resultados
    with open('../resultados/pruebas_estacionariedad.txt', 'w') as f:
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
    with open('../resultados/prueba_cointegracion.txt', 'w') as f:
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
    plt.savefig('../visualizaciones/descomposicion_iva.png', dpi=300)
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
    plt.savefig('../visualizaciones/descomposicion_pib.png', dpi=300)
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
    
    componentes.to_csv('../resultados/componentes_series.csv')
    
    print("Descomposición de series completada.")
    return descomp_iva, descomp_pib

# Función para modelar series temporales con ARIMA/SARIMA
def modelar_series(df_conjunto):
    """
    Modela las series temporales utilizando modelos ARIMA/SARIMA
    """
    print("Modelando series temporales...")
    
    # Preparar datos para modelado
    # Usamos datos hasta 2023 para entrenamiento y 2024 para validación
    fecha_corte = '2023-12-31'
    
    train_iva = df_conjunto.loc[:fecha_corte, 'iva']
    test_iva = df_conjunto.loc[fecha_corte:, 'iva']
    
    # Asegurar que train_iva sea numérico y no tenga NaNs
    train_iva = pd.to_numeric(train_iva, errors='coerce').dropna()
    
    # Visualizar ACF y PACF para IVA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train_iva, ax=ax1, lags=36)
    ax1.set_title('Función de Autocorrelación (ACF) - IVA', fontweight='bold')
    
    plot_pacf(train_iva, ax=ax2, lags=36)
    ax2.set_title('Función de Autocorrelación Parcial (PACF) - IVA', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/acf_pacf_iva.png', dpi=300)
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
        with open('../resultados/resumen_modelo_sarima_iva.txt', 'w') as f:
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
        plt.savefig('../visualizaciones/modelo_sarima_iva.png', dpi=300)
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
            with open('../resultados/metricas_modelo_sarima_iva.txt', 'w') as f:
                f.write("EVALUACIÓN DEL MODELO SARIMA PARA IVA\n")
                f.write("=====================================\n\n")
                f.write(f"RMSE: {rmse_iva:.2f}\n")
                f.write(f"MAE: {mae_iva:.2f}\n")
                f.write(f"MAPE: {mape_iva:.2f}%\n")
        else:
            print("No hay suficientes datos para evaluar el modelo")
            with open('../resultados/metricas_modelo_sarima_iva.txt', 'w') as f:
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
        plt.savefig('../visualizaciones/diagnostico_residuos_sarima.png', dpi=300)
        plt.close(fig)
        
        # Diagnóstico de residuos
        fig = resultado_sarima_iva.plot_diagnostics(figsize=(15, 12))
        fig.suptitle('Diagnóstico de Residuos del Modelo SARIMA para IVA', fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('../visualizaciones/diagnostico_residuos_sarima.png', dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Error en el modelado SARIMA: {e}")
        # Guardar error en archivo
        with open('../resultados/error_modelado.txt', 'w') as f:
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
                        with open('../resultados/causalidad_granger.txt', 'w') as f:
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
                        with open('../resultados/causalidad_granger.txt', 'w') as f:
                            f.write("No hay suficientes datos para la prueba de Granger después de la limpieza.")
                else:
                    print("No hay suficientes datos para la prueba de Granger después de la conversión y limpieza.")
                    with open('../resultados/causalidad_granger.txt', 'w') as f:
                        f.write("No hay suficientes datos para la prueba de Granger después de la conversión y limpieza.")

            except Exception as e_granger:
                print(f"Error en la prueba de causalidad de Granger: {e_granger}")
                with open('../resultados/causalidad_granger.txt', 'w') as f:
                    f.write(f"Error en la prueba de causalidad de Granger: {e_granger}\n")
        else:
            print("No hay suficientes datos o datos vacíos para la prueba de Granger.")
            with open('../resultados/causalidad_granger.txt', 'w') as f:
                f.write("No hay suficientes datos o datos vacíos para la prueba de Granger.\n")
    else:
        print("Modelado SARIMA falló, omitiendo prueba de causalidad de Granger.")
        with open('../resultados/causalidad_granger.txt', 'w') as f:
            f.write("Modelado SARIMA falló, omitiendo prueba de causalidad de Granger.\n")

    print("Modelado de series temporales completado.")
    return resultado_sarima_iva

# Función principal
def main():
    """Función principal para el análisis de series temporales"""
    print("Iniciando análisis de series temporales: PIB e IVA en Colombia (2000-2024)...")
    
    # Cargar datos alineados
    df_conjunto = cargar_datos_alineados()
    
    # Realizar análisis exploratorio
    df_conjunto = realizar_eda(df_conjunto)
    
    # Realizar pruebas de estacionariedad
    df_conjunto = pruebas_estacionariedad(df_conjunto)
    
    # Descomponer series temporales
    descomp_iva, descomp_pib = descomponer_series(df_conjunto)
    
    # Modelar series temporales
    modelo_sarima = modelar_series(df_conjunto)
    
    print("\nAnálisis de series temporales completado con éxito.")
    print("Resultados guardados en las carpetas 'resultados' y 'visualizaciones'.")

if __name__ == "__main__":
    main()
