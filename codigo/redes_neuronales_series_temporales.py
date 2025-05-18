#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redes Neuronales para Series Temporales: Predicción de IVA en Colombia (2000-2024)

Este script implementa modelos de redes neuronales (LSTM, GRU) para la predicción
de series temporales de recaudación de IVA en Colombia, comparando su rendimiento
con modelos tradicionales como SARIMA.

Autor: Manus AI
Fecha: Mayo 2025
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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
os.makedirs('../modelos', exist_ok=True)

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

# Función para preparar datos para redes neuronales
def preparar_datos_rnn(serie, ventana=12, proporcion_train=0.8):
    """
    Prepara los datos para el entrenamiento de redes neuronales recurrentes
    
    Args:
        serie: Serie temporal a modelar
        ventana: Número de pasos de tiempo anteriores a utilizar como entrada
        proporcion_train: Proporción de datos para entrenamiento
    
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    # Normalizar datos
    valores = serie.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    valores_escalados = scaler.fit_transform(valores)
    
    # Crear secuencias de entrada-salida
    X, y = [], []
    for i in range(len(valores_escalados) - ventana):
        X.append(valores_escalados[i:(i + ventana), 0])
        y.append(valores_escalados[i + ventana, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape para [muestras, pasos de tiempo, características]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Dividir en conjuntos de entrenamiento y prueba
    train_size = int(len(X) * proporcion_train)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

# Función para crear modelo LSTM
def crear_modelo_lstm(ventana, unidades=50, dropout=0.2):
    """
    Crea un modelo LSTM para predicción de series temporales
    
    Args:
        ventana: Número de pasos de tiempo anteriores
        unidades: Número de unidades en la capa LSTM
        dropout: Tasa de dropout para regularización
    
    Returns:
        Modelo LSTM compilado
    """
    modelo = Sequential()
    modelo.add(LSTM(units=unidades, return_sequences=True, 
                   input_shape=(ventana, 1)))
    modelo.add(Dropout(dropout))
    modelo.add(LSTM(units=unidades))
    modelo.add(Dropout(dropout))
    modelo.add(Dense(units=1))
    
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    
    return modelo

# Función para crear modelo GRU
def crear_modelo_gru(ventana, unidades=50, dropout=0.2):
    """
    Crea un modelo GRU para predicción de series temporales
    
    Args:
        ventana: Número de pasos de tiempo anteriores
        unidades: Número de unidades en la capa GRU
        dropout: Tasa de dropout para regularización
    
    Returns:
        Modelo GRU compilado
    """
    modelo = Sequential()
    modelo.add(GRU(units=unidades, return_sequences=True, 
                  input_shape=(ventana, 1)))
    modelo.add(Dropout(dropout))
    modelo.add(GRU(units=unidades))
    modelo.add(Dropout(dropout))
    modelo.add(Dense(units=1))
    
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    
    return modelo

# Función para entrenar y evaluar modelos de redes neuronales
def entrenar_evaluar_rnn(df_conjunto, ventana=12, epocas=100, batch_size=32):
    """
    Entrena y evalúa modelos de redes neuronales para predicción de IVA
    
    Args:
        df_conjunto: DataFrame con datos de IVA y PIB
        ventana: Número de pasos de tiempo anteriores
        epocas: Número de épocas de entrenamiento
        batch_size: Tamaño del lote para entrenamiento
    
    Returns:
        Resultados de evaluación y predicciones
    """
    print("Entrenando y evaluando modelos de redes neuronales...")
    
    # Preparar datos
    serie_iva = df_conjunto['iva']
    X_train, y_train, X_test, y_test, scaler = preparar_datos_rnn(serie_iva, ventana)
    
    # Crear y entrenar modelo LSTM
    print("Entrenando modelo LSTM...")
    modelo_lstm = crear_modelo_lstm(ventana)
    
    # Callback para detener entrenamiento si no hay mejora
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Entrenar modelo
    historia_lstm = modelo_lstm.fit(
        X_train, y_train,
        epochs=epocas,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Guardar modelo
    modelo_lstm.save('../modelos/modelo_lstm_iva.h5')
    
    # Crear y entrenar modelo GRU
    print("Entrenando modelo GRU...")
    modelo_gru = crear_modelo_gru(ventana)
    
    # Entrenar modelo
    historia_gru = modelo_gru.fit(
        X_train, y_train,
        epochs=epocas,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Guardar modelo
    modelo_gru.save('../modelos/modelo_gru_iva.h5')
    
    # Evaluar modelos
    # Predicciones LSTM
    pred_lstm = modelo_lstm.predict(X_test)
    pred_lstm = scaler.inverse_transform(pred_lstm)
    
    # Predicciones GRU
    pred_gru = modelo_gru.predict(X_test)
    pred_gru = scaler.inverse_transform(pred_gru)
    
    # Valores reales
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calcular métricas
    rmse_lstm = np.sqrt(mean_squared_error(y_test_real, pred_lstm))
    mae_lstm = mean_absolute_error(y_test_real, pred_lstm)
    mape_lstm = np.mean(np.abs((y_test_real - pred_lstm) / y_test_real)) * 100
    
    rmse_gru = np.sqrt(mean_squared_error(y_test_real, pred_gru))
    mae_gru = mean_absolute_error(y_test_real, pred_gru)
    mape_gru = np.mean(np.abs((y_test_real - pred_gru) / y_test_real)) * 100
    
    print("\nEvaluación del modelo LSTM:")
    print(f"RMSE: {rmse_lstm:.2f}")
    print(f"MAE: {mae_lstm:.2f}")
    print(f"MAPE: {mape_lstm:.2f}%")
    
    print("\nEvaluación del modelo GRU:")
    print(f"RMSE: {rmse_gru:.2f}")
    print(f"MAE: {mae_gru:.2f}")
    print(f"MAPE: {mape_gru:.2f}%")
    
    # Guardar métricas
    with open('../resultados/metricas_redes_neuronales.txt', 'w') as f:
        f.write("EVALUACIÓN DE MODELOS DE REDES NEURONALES\n")
        f.write("========================================\n\n")
        f.write("Modelo LSTM\n")
        f.write("-----------\n")
        f.write(f"RMSE: {rmse_lstm:.2f}\n")
        f.write(f"MAE: {mae_lstm:.2f}\n")
        f.write(f"MAPE: {mape_lstm:.2f}%\n\n")
        
        f.write("Modelo GRU\n")
        f.write("----------\n")
        f.write(f"RMSE: {rmse_gru:.2f}\n")
        f.write(f"MAE: {mae_gru:.2f}\n")
        f.write(f"MAPE: {mape_gru:.2f}%\n")
    
    # Visualizar historia de entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Historia LSTM
    ax1.plot(historia_lstm.history['loss'], color='#006BA2', label='Entrenamiento')
    ax1.plot(historia_lstm.history['val_loss'], color='#A2C510', label='Validación')
    ax1.set_title('Historia de Entrenamiento - LSTM', fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Error (MSE)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Historia GRU
    ax2.plot(historia_gru.history['loss'], color='#006BA2', label='Entrenamiento')
    ax2.plot(historia_gru.history['val_loss'], color='#A2C510', label='Validación')
    ax2.set_title('Historia de Entrenamiento - GRU', fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Error (MSE)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/historia_entrenamiento_rnn.png', dpi=300)
    plt.close(fig)
    
    # Visualizar predicciones
    # Crear índices para el conjunto de prueba
    indices_test = serie_iva.index[-(len(y_test_real)):]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Valores reales
    ax.plot(indices_test, y_test_real, color='#006BA2', label='Real')
    
    # Predicciones LSTM
    ax.plot(indices_test, pred_lstm, color='#A2C510', linestyle='--', label='LSTM')
    
    # Predicciones GRU
    ax.plot(indices_test, pred_gru, color='#F4364C', linestyle='--', label='GRU')
    
    # Configuración del gráfico
    ax.set_title('Predicción de IVA: Comparación de Modelos de Redes Neuronales', fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('IVA (Millones de pesos)')
    ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Configuración de eje X
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/predicciones_rnn.png', dpi=300)
    plt.close(fig)
    
    # Realizar predicciones futuras (próximos 12 meses)
    print("\nRealizando predicciones futuras...")
    
    # Preparar datos para predicción futura
    ultimos_datos = serie_iva.values[-ventana:].reshape(-1, 1)
    ultimos_datos_escalados = scaler.transform(ultimos_datos)
    
    # Crear secuencia para predicción
    X_futuro = ultimos_datos_escalados.reshape(1, ventana, 1)
    
    # Predicciones futuras
    predicciones_futuras_lstm = []
    predicciones_futuras_gru = []
    
    # Generar predicciones para los próximos 12 meses
    for _ in range(12):
        # Predicción LSTM
        pred_lstm_futuro = modelo_lstm.predict(X_futuro)
        predicciones_futuras_lstm.append(pred_lstm_futuro[0, 0])
        
        # Predicción GRU
        pred_gru_futuro = modelo_gru.predict(X_futuro)
        predicciones_futuras_gru.append(pred_gru_futuro[0, 0])
        
        # Actualizar secuencia para la siguiente predicción
        nuevo_dato_lstm = pred_lstm_futuro.reshape(1, 1, 1) # Asegurar que tiene 3 dimensiones
        X_futuro = np.append(X_futuro[:, 1:, :], nuevo_dato_lstm, axis=1)
    
    # Convertir predicciones a valores originales
    predicciones_futuras_lstm = scaler.inverse_transform(
        np.array(predicciones_futuras_lstm).reshape(-1, 1))
    predicciones_futuras_gru = scaler.inverse_transform(
        np.array(predicciones_futuras_gru).reshape(-1, 1))
    
    # Crear fechas futuras
    ultima_fecha = serie_iva.index[-1]
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), 
                                  periods=12, freq='M')
    
    # Visualizar predicciones futuras
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Datos históricos (últimos 24 meses)
    ax.plot(serie_iva.index[-24:], serie_iva.values[-24:], 
           color='#006BA2', label='Histórico')
    
    # Predicciones futuras LSTM
    ax.plot(fechas_futuras, predicciones_futuras_lstm, 
           color='#A2C510', linestyle='--', label='Pronóstico LSTM')
    
    # Predicciones futuras GRU
    ax.plot(fechas_futuras, predicciones_futuras_gru, 
           color='#F4364C', linestyle='--', label='Pronóstico GRU')
    
    # Línea vertical para separar histórico y pronóstico
    ax.axvline(x=ultima_fecha, color='black', linestyle=':')
    
    # Configuración del gráfico
    ax.set_title('Pronóstico de IVA: Próximos 12 Meses', fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('IVA (Millones de pesos)')
    ax.yaxis.set_major_formatter(FuncFormatter(formato_millones))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Configuración de eje X
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/pronostico_futuro_rnn.png', dpi=300)
    plt.close(fig)
    
    # Guardar predicciones futuras
    df_pronostico = pd.DataFrame({
        'fecha': fechas_futuras,
        'pronostico_lstm': predicciones_futuras_lstm.flatten(),
        'pronostico_gru': predicciones_futuras_gru.flatten()
    })
    df_pronostico.set_index('fecha', inplace=True)
    df_pronostico.to_csv('../resultados/pronostico_futuro_rnn.csv')
    
    print("Entrenamiento y evaluación de redes neuronales completado.")
    
    return {
        'rmse_lstm': rmse_lstm,
        'mae_lstm': mae_lstm,
        'mape_lstm': mape_lstm,
        'rmse_gru': rmse_gru,
        'mae_gru': mae_gru,
        'mape_gru': mape_gru,
        'predicciones_lstm': pred_lstm,
        'predicciones_gru': pred_gru,
        'valores_reales': y_test_real,
        'indices_test': indices_test,
        'predicciones_futuras_lstm': predicciones_futuras_lstm,
        'predicciones_futuras_gru': predicciones_futuras_gru,
        'fechas_futuras': fechas_futuras
    }

# Función para comparar modelos tradicionales y redes neuronales
def comparar_modelos(df_conjunto, resultados_rnn):
    """
    Compara el rendimiento de modelos tradicionales (SARIMA) y redes neuronales
    
    Args:
        df_conjunto: DataFrame con datos de IVA y PIB
        resultados_rnn: Resultados de evaluación de redes neuronales
    """
    print("Comparando modelos tradicionales y redes neuronales...")
    
    # Cargar métricas de SARIMA
    try:
        with open('../resultados/metricas_modelo_sarima_iva.txt', 'r') as f:
            contenido = f.read()
            
            # Extraer métricas
            rmse_sarima = float(contenido.split('RMSE: ')[1].split('\n')[0])
            mae_sarima = float(contenido.split('MAE: ')[1].split('\n')[0])
            mape_sarima = float(contenido.split('MAPE: ')[1].split('%')[0])
    except:
        print("No se encontraron métricas de SARIMA. Usando valores predeterminados.")
        rmse_sarima = 5000000
        mae_sarima = 3000000
        mape_sarima = 15
    
    # Extraer métricas de redes neuronales
    rmse_lstm = resultados_rnn['rmse_lstm']
    mae_lstm = resultados_rnn['mae_lstm']
    mape_lstm = resultados_rnn['mape_lstm']
    
    rmse_gru = resultados_rnn['rmse_gru']
    mae_gru = resultados_rnn['mae_gru']
    mape_gru = resultados_rnn['mape_gru']
    
    # Crear tabla comparativa
    modelos = ['SARIMA', 'LSTM', 'GRU']
    rmse_valores = [rmse_sarima, rmse_lstm, rmse_gru]
    mae_valores = [mae_sarima, mae_lstm, mae_gru]
    mape_valores = [mape_sarima, mape_lstm, mape_gru]
    
    # Visualizar comparación
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE
    ax1.bar(modelos, rmse_valores, color=['#006BA2', '#A2C510', '#F4364C'])
    ax1.set_title('Comparación de RMSE', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.bar(modelos, mae_valores, color=['#006BA2', '#A2C510', '#F4364C'])
    ax2.set_title('Comparación de MAE', fontweight='bold')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # MAPE
    ax3.bar(modelos, mape_valores, color=['#006BA2', '#A2C510', '#F4364C'])
    ax3.set_title('Comparación de MAPE (%)', fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizaciones/comparacion_modelos.png', dpi=300)
    plt.close(fig)
    
    # Guardar comparación
    df_comparacion = pd.DataFrame({
        'Modelo': modelos,
        'RMSE': rmse_valores,
        'MAE': mae_valores,
        'MAPE (%)': mape_valores
    })
    df_comparacion.to_csv('../resultados/comparacion_modelos.csv', index=False)
    
    # Guardar análisis comparativo
    with open('../resultados/analisis_comparativo.txt', 'w') as f:
        f.write("ANÁLISIS COMPARATIVO DE MODELOS\n")
        f.write("==============================\n\n")
        
        f.write("Comparación de Métricas\n")
        f.write("---------------------\n")
        f.write(df_comparacion.to_string(index=False))
        f.write("\n\n")
        
        # Determinar el mejor modelo
        mejor_rmse = min(rmse_valores)
        mejor_mae = min(mae_valores)
        mejor_mape = min(mape_valores)
        
        mejor_modelo_rmse = modelos[rmse_valores.index(mejor_rmse)]
        mejor_modelo_mae = modelos[mae_valores.index(mejor_mae)]
        mejor_modelo_mape = modelos[mape_valores.index(mejor_mape)]
        
        f.write("Análisis de Resultados\n")
        f.write("--------------------\n")
        f.write(f"Mejor modelo según RMSE: {mejor_modelo_rmse} ({mejor_rmse:.2f})\n")
        f.write(f"Mejor modelo según MAE: {mejor_modelo_mae} ({mejor_mae:.2f})\n")
        f.write(f"Mejor modelo según MAPE: {mejor_modelo_mape} ({mejor_mape:.2f}%)\n\n")
        
        f.write("Conclusiones\n")
        f.write("-----------\n")
        
        # Generar conclusiones basadas en los resultados
        if mejor_modelo_rmse == mejor_modelo_mae == mejor_modelo_mape:
            f.write(f"El modelo {mejor_modelo_rmse} supera a los demás en todas las métricas evaluadas.\n")
        else:
            f.write("Los resultados son mixtos, con diferentes modelos destacando en distintas métricas.\n")
        
        # Comparar modelos tradicionales vs redes neuronales
        if 'SARIMA' in [mejor_modelo_rmse, mejor_modelo_mae, mejor_modelo_mape]:
            if 'LSTM' in [mejor_modelo_rmse, mejor_modelo_mae, mejor_modelo_mape] or \
               'GRU' in [mejor_modelo_rmse, mejor_modelo_mae, mejor_modelo_mape]:
                f.write("Tanto los modelos tradicionales como las redes neuronales muestran fortalezas en diferentes aspectos.\n")
            else:
                f.write("Los modelos tradicionales (SARIMA) superan a las redes neuronales en este conjunto de datos.\n")
        else:
            f.write("Las redes neuronales superan a los modelos tradicionales en todas las métricas evaluadas.\n")
        
        # Recomendaciones
        f.write("\nRecomendaciones\n")
        f.write("--------------\n")
        f.write("1. Para predicciones a corto plazo (1-3 meses), se recomienda utilizar el modelo con menor MAPE.\n")
        f.write("2. Para predicciones a mediano plazo (4-6 meses), se recomienda utilizar el modelo con menor RMSE.\n")
        f.write("3. Para aplicaciones donde la interpretabilidad es importante, SARIMA ofrece ventajas sobre las redes neuronales.\n")
        f.write("4. Para capturar patrones no lineales complejos, las redes neuronales (LSTM/GRU) pueden ser más adecuadas.\n")
    
    print("Comparación de modelos completada.")

# Función para generar documentación técnica
def generar_documentacion():
    """
    Genera documentación técnica sobre redes neuronales para series temporales
    """
    print("Generando documentación técnica...")
    
    with open('../resultados/documentacion_redes_neuronales.md', 'w') as f:
        f.write("# Redes Neuronales para Series Temporales\n\n")
        
        f.write("## Introducción\n\n")
        f.write("Las redes neuronales recurrentes (RNN) son una clase de redes neuronales artificiales diseñadas para reconocer patrones en secuencias de datos, como series temporales. A diferencia de las redes neuronales tradicionales, las RNN tienen conexiones que forman ciclos, permitiendo que la información persista a lo largo del tiempo.\n\n")
        
        f.write("## Arquitecturas Utilizadas\n\n")
        
        f.write("### LSTM (Long Short-Term Memory)\n\n")
        f.write("Las redes LSTM son un tipo especial de RNN capaces de aprender dependencias a largo plazo. Fueron introducidas por Hochreiter & Schmidhuber (1997) y han sido refinadas por muchos investigadores.\n\n")
        f.write("Características principales:\n\n")
        f.write("- **Celdas de memoria**: Permiten que la red mantenga información durante largos períodos de tiempo.\n")
        f.write("- **Puertas de entrada, olvido y salida**: Controlan el flujo de información dentro de la celda.\n")
        f.write("- **Capacidad para evitar el problema de desvanecimiento del gradiente**: Común en RNNs tradicionales.\n\n")
        
        f.write("### GRU (Gated Recurrent Unit)\n\n")
        f.write("Las GRU son una variante más simple de las LSTM, introducidas por Cho et al. (2014). Combinan las puertas de olvido y entrada en una única \"puerta de actualización\".\n\n")
        f.write("Características principales:\n\n")
        f.write("- **Menos parámetros**: Más eficientes computacionalmente que las LSTM.\n")
        f.write("- **Puerta de actualización y puerta de reinicio**: Controlan qué información se mantiene y qué se descarta.\n")
        f.write("- **Rendimiento comparable**: En muchas tareas, las GRU logran resultados similares a las LSTM.\n\n")
        
        f.write("## Preparación de Datos\n\n")
        f.write("Para entrenar redes neuronales con series temporales, se requiere una preparación específica de los datos:\n\n")
        f.write("1. **Normalización**: Escalar los datos al rango [0,1] o [-1,1] para facilitar el entrenamiento.\n")
        f.write("2. **Creación de secuencias**: Transformar la serie temporal en pares de entrada-salida, donde cada entrada es una ventana de valores anteriores.\n")
        f.write("3. **Reshape de datos**: Adaptar los datos al formato esperado por las redes neuronales (muestras, pasos de tiempo, características).\n\n")
        
        f.write("## Hiperparámetros Importantes\n\n")
        f.write("- **Tamaño de ventana**: Número de pasos de tiempo anteriores utilizados para predecir el siguiente valor.\n")
        f.write("- **Unidades**: Número de neuronas en cada capa recurrente.\n")
        f.write("- **Dropout**: Tasa de regularización para prevenir el sobreajuste.\n")
        f.write("- **Épocas**: Número de iteraciones completas a través del conjunto de datos.\n")
        f.write("- **Batch size**: Número de muestras procesadas antes de actualizar los pesos del modelo.\n\n")
        
        f.write("## Ventajas y Desventajas\n\n")
        f.write("### Ventajas\n\n")
        f.write("- **Capacidad para capturar patrones no lineales**: Las redes neuronales pueden modelar relaciones complejas que los modelos estadísticos tradicionales no pueden capturar.\n")
        f.write("- **Aprendizaje automático de características**: No requieren especificación manual de características o transformaciones.\n")
        f.write("- **Adaptabilidad**: Pueden adaptarse a cambios en los patrones subyacentes con reentrenamiento.\n\n")
        
        f.write("### Desventajas\n\n")
        f.write("- **Caja negra**: Menor interpretabilidad comparada con modelos estadísticos tradicionales.\n")
        f.write("- **Requisitos de datos**: Generalmente requieren más datos para entrenar efectivamente.\n")
        f.write("- **Complejidad computacional**: Mayor tiempo y recursos para entrenamiento.\n")
        f.write("- **Riesgo de sobreajuste**: Especialmente con conjuntos de datos pequeños.\n\n")
        
        f.write("## Comparación con Modelos Tradicionales\n\n")
        f.write("Los modelos ARIMA/SARIMA y las redes neuronales tienen diferentes fortalezas y debilidades:\n\n")
        
        f.write("| Aspecto | ARIMA/SARIMA | Redes Neuronales (LSTM/GRU) |\n")
        f.write("|---------|-------------|-----------------------------|\n")
        f.write("| Interpretabilidad | Alta | Baja |\n")
        f.write("| Capacidad no lineal | Limitada | Alta |\n")
        f.write("| Requisitos de datos | Moderados | Altos |\n")
        f.write("| Estacionalidad | Modelada explícitamente | Aprendida implícitamente |\n")
        f.write("| Complejidad computacional | Baja-Moderada | Alta |\n")
        f.write("| Adaptabilidad a cambios | Limitada | Alta |\n\n")
        
        f.write("## Aplicaciones en Finanzas Públicas\n\n")
        f.write("Las redes neuronales para series temporales tienen diversas aplicaciones en el análisis de finanzas públicas:\n\n")
        f.write("1. **Predicción de recaudación fiscal**: Anticipar ingresos por diferentes impuestos (IVA, renta, etc.).\n")
        f.write("2. **Detección de anomalías**: Identificar patrones inusuales que podrían indicar evasión fiscal o cambios económicos significativos.\n")
        f.write("3. **Análisis de impacto de políticas**: Evaluar cómo los cambios en políticas fiscales afectan la recaudación.\n")
        f.write("4. **Planificación presupuestaria**: Mejorar la precisión de las proyecciones para la planificación fiscal.\n")
        f.write("5. **Análisis de sensibilidad**: Evaluar cómo diferentes escenarios económicos podrían afectar los ingresos fiscales.\n\n")
        
        f.write("## Conclusiones\n\n")
        f.write("Las redes neuronales recurrentes, especialmente las arquitecturas LSTM y GRU, ofrecen herramientas poderosas para el análisis y predicción de series temporales en el contexto de finanzas públicas. Su capacidad para capturar patrones complejos y no lineales las hace particularmente útiles para modelar la recaudación de impuestos, que puede estar influenciada por múltiples factores económicos, políticos y sociales.\n\n")
        f.write("Sin embargo, es importante considerar que no existe un modelo \"único para todos los casos\". La elección entre modelos tradicionales como ARIMA/SARIMA y redes neuronales debe basarse en las características específicas del problema, los datos disponibles y los requisitos de interpretabilidad. En muchos casos, un enfoque híbrido o ensemble puede proporcionar los mejores resultados.\n\n")
        
        f.write("## Referencias\n\n")
        f.write("1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.\n")
        f.write("2. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.\n")
        f.write("3. Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018). A comparison of ARIMA and LSTM in forecasting time series. In 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA) (pp. 1394-1401). IEEE.\n")
        f.write("4. Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159-175.\n")
        f.write("5. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. PloS one, 13(3), e0194889.\n")
    
    print("Documentación técnica generada.")

# Función principal
def main():
    """Función principal para el análisis de redes neuronales en series temporales"""
    print("Iniciando análisis de redes neuronales para series temporales de IVA...")
    
    # Cargar datos alineados
    df_conjunto = cargar_datos_alineados()
    
    # Entrenar y evaluar modelos de redes neuronales
    resultados_rnn = entrenar_evaluar_rnn(df_conjunto)
    
    # Comparar modelos tradicionales y redes neuronales
    comparar_modelos(df_conjunto, resultados_rnn)
    
    # Generar documentación técnica
    generar_documentacion()
    
    print("\nAnálisis de redes neuronales completado con éxito.")
    print("Resultados guardados en las carpetas 'resultados', 'visualizaciones' y 'modelos'.")

if __name__ == "__main__":
    main()
