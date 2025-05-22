#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estandarización y Alineación de Series Temporales: PIB e IVA en Colombia (2000-2024)

Este script estandariza y alinea las series temporales de PIB e IVA
para asegurar que ambas series tengan las mismas fechas y estén correctamente
preparadas para el análisis conjunto.

Autor: Manus AI
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

# --- Definición de rutas absolutas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATOS_DIR = os.path.join(PROJECT_ROOT, 'datos')
VISUALIZACIONES_DIR = os.path.join(PROJECT_ROOT, 'visualizaciones')
RESULTADOS_DIR = os.path.join(PROJECT_ROOT, 'resultados')
# --- Fin Definición de rutas absolutas ---

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
os.makedirs(DATOS_DIR, exist_ok=True)
os.makedirs(VISUALIZACIONES_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True) # Aseguramos que la carpeta resultados también use la ruta absoluta

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
    # Ruta corregida para leer desde la carpeta de datos del proyecto
    ruta_archivo_iva = os.path.join(DATOS_DIR, 'data_iva.xlsx')
    
    if not os.path.exists(ruta_archivo_iva):
        print(f"ERROR CRÍTICO: No se encontró el archivo de datos del IVA en la ruta: {ruta_archivo_iva}")
        print("Por favor, asegúrese de que el archivo 'data_iva.xlsx' se encuentre en la carpeta 'datos' del proyecto.")
        return None # Retornar None si el archivo no existe

    df_iva = pd.read_excel(ruta_archivo_iva)
    
    # Mostrar las primeras filas para entender la estructura
    print("Primeras filas de datos IVA:")
    print(df_iva.head())
    
    # Asegurar que la columna de fecha sea de tipo datetime
    df_iva['fecha'] = pd.to_datetime(df_iva['fecha'])
    
    # Renombrar columnas para mayor claridad
    df_iva.rename(columns={'valor': 'iva'}, inplace=True)
    
    # Crear columnas de año y mes para facilitar análisis
    df_iva['año'] = df_iva['fecha'].dt.year
    df_iva['mes'] = df_iva['fecha'].dt.month
    
    # Ordenar por fecha
    df_iva = df_iva.sort_values('fecha')
    df_iva.set_index('fecha', inplace=True)
    
    print(f"Datos de IVA cargados: {len(df_iva)} registros mensuales")
    print(f"Periodo: {df_iva.index.min().strftime('%Y-%m')} a {df_iva.index.max().strftime('%Y-%m')}")

    # Visualizar serie de IVA
    fig_iva, ax_iva = plt.subplots(figsize=(12, 6))
    ax_iva.plot(df_iva.index, df_iva['iva'], color='#006BA2', linewidth=2)
    ax_iva.set_title('Recaudación de IVA en Colombia (Datos Crudos)', fontweight='bold')
    ax_iva.set_ylabel('IVA (Millones de pesos)')
    ax_iva.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    ax_iva.grid(True, alpha=0.3)
    ax_iva.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_iva.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'iva_datos_crudos.png'), dpi=300)
    plt.close(fig_iva)
    print(f"Visualización de datos crudos de IVA guardada en {os.path.join(VISUALIZACIONES_DIR, 'iva_datos_crudos.png')}")
    
    return df_iva

def cargar_datos_pib():
    """
    Carga los datos de PIB en diferentes frecuencias
    """
    print("Cargando datos de PIB...")
    
    # Cargar datos anuales
    df_pib_anual = pd.read_csv(os.path.join(DATOS_DIR, 'pib_anual_colombia.csv'))
    
    # Cargar datos trimestrales
    df_pib_trimestral = pd.read_csv(os.path.join(DATOS_DIR, 'pib_trimestral_colombia.csv'))
    
    # Cargar datos mensuales
    df_pib_mensual = pd.read_csv(os.path.join(DATOS_DIR, 'pib_mensual_colombia.csv'))
    
    # Mostrar las primeras filas para entender la estructura
    print("Primeras filas de datos PIB mensual:")
    print(df_pib_mensual.head())
    
    # Convertir columnas de fecha para datos mensuales
    df_pib_mensual['fecha'] = pd.to_datetime(df_pib_mensual.apply(
        lambda x: f"{int(x['año'])}-{int(x['mes']):02d}-01", axis=1))
    df_pib_mensual.set_index('fecha', inplace=True)
    
    print(f"Datos de PIB cargados:")
    print(f"- Anuales: {len(df_pib_anual)} registros")
    print(f"- Trimestrales: {len(df_pib_trimestral)} registros")
    print(f"- Mensuales: {len(df_pib_mensual)} registros")
    print(f"Periodo mensual: {df_pib_mensual.index.min().strftime('%Y-%m')} a {df_pib_mensual.index.max().strftime('%Y-%m')}")

    # Visualizar serie de PIB mensual
    fig_pib, ax_pib = plt.subplots(figsize=(12, 6))
    ax_pib.plot(df_pib_mensual.index, df_pib_mensual['pib_usd'], color='#A2C510', linewidth=2)
    ax_pib.set_title('PIB Mensual de Colombia (Datos Crudos)', fontweight='bold')
    ax_pib.set_ylabel('PIB (Miles de millones USD)')
    ax_pib.yaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
    ax_pib.grid(True, alpha=0.3)
    ax_pib.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_pib.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'pib_mensual_datos_crudos.png'), dpi=300)
    plt.close(fig_pib)
    print(f"Visualización de datos crudos de PIB mensual guardada en {os.path.join(VISUALIZACIONES_DIR, 'pib_mensual_datos_crudos.png')}")
    
    return df_pib_anual, df_pib_trimestral, df_pib_mensual

def estandarizar_y_alinear_series(df_iva, df_pib_mensual):
    """
    Estandariza y alinea las series temporales de IVA y PIB para asegurar compatibilidad
    """
    print("Estandarizando y alineando series temporales de IVA y PIB...")
    
    # Crear copias para no modificar los originales
    iva = df_iva.copy()
    pib = df_pib_mensual.copy()
    
    # Estandarizar fechas: convertir ambas series al primer día del mes
    print("Estandarizando fechas al primer día de cada mes...")
    
    # Para IVA: convertir al primer día del mes
    iva['fecha_estandar'] = iva.index.to_period('M').to_timestamp()
    
    # Para PIB: ya está en el primer día del mes, pero aseguramos consistencia
    pib['fecha_estandar'] = pib.index.to_period('M').to_timestamp()
    
    # Verificar rangos de fechas estandarizadas
    print(f"Rango de fechas IVA estandarizadas: {iva['fecha_estandar'].min().strftime('%Y-%m-%d')} a {iva['fecha_estandar'].max().strftime('%Y-%m-%d')}")
    print(f"Rango de fechas PIB estandarizadas: {pib['fecha_estandar'].min().strftime('%Y-%m-%d')} a {pib['fecha_estandar'].max().strftime('%Y-%m-%d')}")
    
    # Encontrar el rango común de fechas
    fecha_inicio = max(iva['fecha_estandar'].min(), pib['fecha_estandar'].min())
    fecha_fin = min(iva['fecha_estandar'].max(), pib['fecha_estandar'].max())
    
    print(f"Rango común de fechas: {fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}")
    
    # Filtrar ambas series al rango común
    iva_filtrado = iva[(iva['fecha_estandar'] >= fecha_inicio) & (iva['fecha_estandar'] <= fecha_fin)]
    pib_filtrado = pib[(pib['fecha_estandar'] >= fecha_inicio) & (pib['fecha_estandar'] <= fecha_fin)]
    
    # Crear DataFrames con fecha estandarizada como índice
    iva_df = iva_filtrado[['fecha_estandar', 'iva']].set_index('fecha_estandar')
    pib_df = pib_filtrado[['fecha_estandar', 'pib_usd']].set_index('fecha_estandar')
    
    # Unir los DataFrames por el índice de fecha estandarizada
    df_conjunto = pd.merge(iva_df, pib_df, left_index=True, right_index=True, how='inner')
    
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
    df_conjunto.to_csv(os.path.join(DATOS_DIR, 'iva_pib_alineado.csv'))
    
    print(f"Series alineadas con éxito: {len(df_conjunto)} registros mensuales")
    
    if len(df_conjunto) > 0:
        print(f"Periodo final: {df_conjunto.index.min().strftime('%Y-%m-%d')} a {df_conjunto.index.max().strftime('%Y-%m-%d')}")
        
        # Visualizar series alineadas
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Gráfico de IVA
        ax1.plot(df_conjunto.index, df_conjunto['iva'], color='#006BA2', linewidth=2)
        ax1.set_title('Recaudación de IVA en Colombia (Series Alineadas)', fontweight='bold')
        ax1.set_ylabel('IVA (Millones de pesos)')
        ax1.yaxis.set_major_formatter(FuncFormatter(formato_millones))
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de PIB
        ax2.plot(df_conjunto.index, df_conjunto['pib_usd'], color='#A2C510', linewidth=2)
        ax2.set_title('PIB de Colombia (Series Alineadas)', fontweight='bold')
        ax2.set_ylabel('PIB (USD)')
        ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
        ax2.grid(True, alpha=0.3)
        
        # Configuración de eje X
        ax2.set_xlabel('Año')
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_alineadas.png'), dpi=300)
        
        # Calcular correlación
        correlacion = df_conjunto['iva'].corr(df_conjunto['pib_usd'])
        print(f"Correlación entre IVA y PIB (series alineadas): {correlacion:.4f}")
    else:
        print("ADVERTENCIA: No hay registros coincidentes entre las series después de la alineación.")
        print("Verificando meses y años disponibles en cada serie...")
        
        # Crear conjuntos de pares (año, mes) para cada serie
        meses_iva = set([(y, m) for y, m in zip(iva['año'], iva['mes'])])
        meses_pib = set([(y, m) for y, m in zip(pib['año'], pib['mes'])])
        
        # Encontrar meses comunes
        meses_comunes = meses_iva.intersection(meses_pib)
        
        print(f"Meses comunes entre series: {len(meses_comunes)}")
        if len(meses_comunes) > 0:
            print("Ejemplos de meses comunes: ", list(meses_comunes)[:5])
            
            # Crear un DataFrame con los meses comunes
            registros_comunes = []
            
            for año, mes in meses_comunes:
                # Encontrar el registro de IVA para este mes
                iva_registro = iva[(iva['año'] == año) & (iva['mes'] == mes)].iloc[0]
                
                # Encontrar el registro de PIB para este mes
                pib_registro = pib[(pib['año'] == año) & (pib['mes'] == mes)].iloc[0]
                
                # Agregar al conjunto de registros comunes
                registros_comunes.append({
                    'fecha': pd.Timestamp(year=año, month=mes, day=1),
                    'iva': iva_registro['iva'],
                    'pib': pib_registro['pib_usd'],
                    'año': año,
                    'mes': mes
                })
            
            # Crear DataFrame con registros comunes
            df_comun = pd.DataFrame(registros_comunes)
            df_comun.set_index('fecha', inplace=True)
            
            # Guardar DataFrame con registros comunes
            df_comun.to_csv(os.path.join(DATOS_DIR, 'iva_pib_alineado.csv'))
            
            print(f"Series alineadas manualmente con éxito: {len(df_comun)} registros mensuales")
            print(f"Periodo final: {df_comun.index.min().strftime('%Y-%m-%d')} a {df_comun.index.max().strftime('%Y-%m-%d')}")
            
            # Visualizar series alineadas manualmente
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Gráfico de IVA
            ax1.plot(df_comun.index, df_comun['iva'], color='#006BA2', linewidth=2)
            ax1.set_title('Recaudación de IVA en Colombia (Series Alineadas Manualmente)', fontweight='bold')
            ax1.set_ylabel('IVA (Millones de pesos)')
            ax1.yaxis.set_major_formatter(FuncFormatter(formato_millones))
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de PIB
            ax2.plot(df_comun.index, df_comun['pib'], color='#A2C510', linewidth=2)
            ax2.set_title('PIB de Colombia (Series Alineadas Manualmente)', fontweight='bold')
            ax2.set_ylabel('PIB (USD)')
            ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles_millones))
            ax2.grid(True, alpha=0.3)
            
            # Configuración de eje X
            ax2.set_xlabel('Año')
            ax2.xaxis.set_major_locator(mdates.YearLocator(2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZACIONES_DIR, 'series_alineadas_manual.png'), dpi=300)
            
            # Calcular correlación
            correlacion = df_comun['iva'].corr(df_comun['pib'])
            print(f"Correlación entre IVA y PIB (series alineadas manualmente): {correlacion:.4f}")
            
            return df_comun
        else:
            print("ERROR: No se encontraron meses comunes entre las series.")
            return None
    
    return df_conjunto

def main():
    """Función principal para estandarizar y alinear series temporales"""
    print("Iniciando estandarización y alineación de series temporales: PIB e IVA en Colombia (2000-2024)...")
    
    # Cargar datos
    df_iva = cargar_datos_iva()
    
    # Si df_iva es None (porque el archivo no se encontró), detener la ejecución o manejar el error.
    if df_iva is None:
        print("\nERROR CRÍTICO EN MAIN: No se pudieron cargar los datos del IVA. Revisar mensajes anteriores.")
        print("La alineación de series no puede continuar.")
        return

    df_pib_anual, df_pib_trimestral, df_pib_mensual = cargar_datos_pib()
    
    # Estandarizar y alinear series
    df_conjunto = estandarizar_y_alinear_series(df_iva, df_pib_mensual)
    
    if df_conjunto is not None and len(df_conjunto) > 0:
        print("\nEstandarización y alineación de series temporales completada con éxito.")
        print(f"DataFrame conjunto guardado en: {os.path.join(DATOS_DIR, 'iva_pib_alineado.csv')}")
        # Determinar si se usó la alineación manual para el mensaje de la visualización
        if os.path.exists(os.path.join(VISUALIZACIONES_DIR, 'series_alineadas_manual.png')) and not os.path.exists(os.path.join(VISUALIZACIONES_DIR, 'series_alineadas.png')):
             print(f"Visualización guardada en: {os.path.join(VISUALIZACIONES_DIR, 'series_alineadas_manual.png')}")
        else:
             print(f"Visualización guardada en: {os.path.join(VISUALIZACIONES_DIR, 'series_alineadas.png')}")
    else:
        print("\nERROR: No se pudo completar la alineación de series temporales.")

if __name__ == "__main__":
    main()
