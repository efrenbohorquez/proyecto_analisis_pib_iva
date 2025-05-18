#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alineación de Series Temporales: PIB e IVA en Colombia (2000-2024)

Este script realiza la alineación de las series temporales de PIB e IVA
para asegurar que ambas series tengan las mismas fechas y estén correctamente
preparadas para el análisis conjunto.


Fecha: Mayo 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings

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
os.makedirs('../datos', exist_ok=True)
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

def cargar_datos_iva():
    """
    Carga y prepara los datos de recaudación de IVA
    """
    print("Cargando datos de IVA...")
    
    # Cargar datos de IVA desde el archivo Excel
    df_iva = pd.read_excel('/home/ubuntu/upload/data_iva.xlsx')
    
    # Asegurar que la columna de fecha sea de tipo datetime
    df_iva['fecha'] = pd.to_datetime(df_iva['fecha'])
    
    # Renombrar columnas para mayor claridad
    df_iva.rename(columns={'valor': 'iva'}, inplace=True)
    
    # Crear columnas de año y mes para facilitar análisis
    df_iva['año'] = df_iva['fecha'].dt.year
    df_iva['mes'] = df_iva['fecha'].dt.month
    
    # Ordenar por fecha
    df_iva = df_iva.sort_values('fecha')
    
    # Establecer fecha como índice
    df_iva.set_index('fecha', inplace=True)
    
    print(f"Datos de IVA cargados: {len(df_iva)} registros mensuales")
    print(f"Periodo: {df_iva.index.min().strftime('%Y-%m')} a {df_iva.index.max().strftime('%Y-%m')}")
    
    return df_iva

def cargar_datos_pib():
    """
    Carga los datos de PIB en diferentes frecuencias
    """
    print("Cargando datos de PIB...")
    
    # Cargar datos anuales
    df_pib_anual = pd.read_csv('../datos/pib_anual_colombia.csv')
    
    # Cargar datos trimestrales
    df_pib_trimestral = pd.read_csv('../datos/pib_trimestral_colombia.csv')
    
    # Cargar datos mensuales
    df_pib_mensual = pd.read_csv('../datos/pib_mensual_colombia.csv')
    
    # Convertir columnas de fecha para datos mensuales
    df_pib_mensual['fecha'] = pd.to_datetime(df_pib_mensual.apply(
        lambda x: f"{int(x['año'])}-{int(x['mes']):02d}-01", axis=1))
    
    # Establecer fecha como índice para datos mensuales
    df_pib_mensual.set_index('fecha', inplace=True)
    
    print(f"Datos de PIB cargados:")
    print(f"- Anuales: {len(df_pib_anual)} registros")
    print(f"- Trimestrales: {len(df_pib_trimestral)} registros")
    print(f"- Mensuales: {len(df_pib_mensual)} registros")
    print(f"Periodo mensual: {df_pib_mensual.index.min().strftime('%Y-%m')} a {df_pib_mensual.index.max().strftime('%Y-%m')}")
    
    return df_pib_anual, df_pib_trimestral, df_pib_mensual

def alinear_series(df_iva, df_pib_mensual):
    """
    Alinea las series temporales de IVA y PIB para asegurar compatibilidad
    """
    print("Alineando series temporales de IVA y PIB...")
    
    # Verificar rangos de fechas
    print(f"Rango de fechas IVA: {df_iva.index.min().strftime('%Y-%m-%d')} a {df_iva.index.max().strftime('%Y-%m-%d')}")
    print(f"Rango de fechas PIB: {df_pib_mensual.index.min().strftime('%Y-%m-%d')} a {df_pib_mensual.index.max().strftime('%Y-%m-%d')}")
    
    # Encontrar el rango común de fechas
    fecha_inicio = max(df_iva.index.min(), df_pib_mensual.index.min())
    fecha_fin = min(df_iva.index.max(), df_pib_mensual.index.max())
    
    print(f"Rango común de fechas: {fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}")
    
    # Filtrar ambas series al rango común
    df_iva_alineado = df_iva.loc[fecha_inicio:fecha_fin].copy()
    df_pib_alineado = df_pib_mensual.loc[fecha_inicio:fecha_fin].copy()
    
    # Verificar que ambas series tienen las mismas fechas
    fechas_iva = set(df_iva_alineado.index)
    fechas_pib = set(df_pib_alineado.index)
    
    if fechas_iva != fechas_pib:
        print("ADVERTENCIA: Las fechas no coinciden exactamente después de la alineación.")
        print(f"Fechas en IVA pero no en PIB: {fechas_iva - fechas_pib}")
        print(f"Fechas en PIB pero no en IVA: {fechas_pib - fechas_iva}")
        
        # Usar solo las fechas comunes
        fechas_comunes = fechas_iva.intersection(fechas_pib)
        df_iva_alineado = df_iva_alineado.loc[df_iva_alineado.index.isin(fechas_comunes)]
        df_pib_alineado = df_pib_alineado.loc[df_pib_alineado.index.isin(fechas_comunes)]
    
    # Crear DataFrame conjunto
    df_conjunto = pd.DataFrame({
        'iva': df_iva_alineado['iva'],
        'pib': df_pib_alineado['pib_usd']
    })
    
    # Verificar que no hay valores faltantes
    valores_faltantes = df_conjunto.isna().sum()
    print(f"Valores faltantes después de alineación:")
    print(valores_faltantes)
    
    if valores_faltantes.sum() > 0:
        print("ADVERTENCIA: Hay valores faltantes después de la alineación.")
        # Eliminar filas con valores faltantes
        df_conjunto = df_conjunto.dropna()
        print(f"DataFrame después de eliminar valores faltantes: {len(df_conjunto)} registros")
    
    # Guardar DataFrame alineado
    df_conjunto.to_csv('../datos/iva_pib_alineado.csv')
    
    print(f"Series alineadas con éxito: {len(df_conjunto)} registros mensuales")
    print(f"Periodo final: {df_conjunto.index.min().strftime('%Y-%m')} a {df_conjunto.index.max().strftime('%Y-%m')}")
    
    # Visualizar series alineadas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Gráfico de IVA
    ax1.plot(df_conjunto.index, df_conjunto['iva'], color='#006BA2', linewidth=2)
    ax1.set_title('Recaudación de IVA en Colombia (Series Alineadas)', fontweight='bold')
    ax1.set_ylabel('IVA (Millones de pesos)')
    ax1.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de PIB
    ax2.plot(df_conjunto.index, df_conjunto['pib'], color='#A2C510', linewidth=2)
    ax2.set_title('PIB de Colombia (Series Alineadas)', fontweight='bold')
    ax2.set_ylabel('PIB (Miles de millones USD)')
    ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
    ax2.grid(True, alpha=0.3)
    
    # Configuración de eje X
    ax2.set_xlabel('Año')
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/series_alineadas.png', dpi=300)
    
    # Calcular correlación
    correlacion = df_conjunto['iva'].corr(df_conjunto['pib'])
    print(f"Correlación entre IVA y PIB (series alineadas): {correlacion:.4f}")
    
    return df_conjunto

def main():
    """Función principal para alinear series temporales"""
    print("Iniciando alineación de series temporales: PIB e IVA en Colombia (2000-2024)...")
    
    # Cargar datos
    df_iva = cargar_datos_iva()
    df_pib_anual, df_pib_trimestral, df_pib_mensual = cargar_datos_pib()
    
    # Alinear series
    df_conjunto = alinear_series(df_iva, df_pib_mensual)
    
    print("\nAlineación de series temporales completada con éxito.")
    print(f"DataFrame conjunto guardado en: ../datos/iva_pib_alineado.csv")
    print(f"Visualización guardada en: ../visualizaciones/series_alineadas.png")

if __name__ == "__main__":
    main()
