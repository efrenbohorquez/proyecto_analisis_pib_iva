import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocesamiento_datos import PreprocesadorPIBIVA
from evaluacion_modelos import EvaluadorModelos
from visualizacion import VisualizadorPIBIVA

class ProyectoPIBIVA:
    """Clase principal que orquestra todo el proyecto"""
    
    def __init__(self):
        self.ruta_base = Path(__file__).parent.parent
        self.ruta_datos = self.ruta_base / "datos"
        self.preparar_directorios()
        
    def preparar_directorios(self):
        """Crear directorios necesarios"""
        self.ruta_datos.mkdir(exist_ok=True)
        (self.ruta_base / "resultados").mkdir(exist_ok=True)
        
    def generar_datos_ejemplo(self):
        """Generar datos de ejemplo si no existen archivos"""
        print("Generando datos de ejemplo...")
        
        # Generar fechas mensuales para 5 años
        fechas = pd.date_range('2019-01-01', '2023-12-31', freq='M')
        
        # PIB simulado con tendencia y estacionalidad
        trend = np.linspace(100000, 120000, len(fechas))
        seasonal = 5000 * np.sin(2 * np.pi * np.arange(len(fechas)) / 12)
        noise = np.random.normal(0, 2000, len(fechas))
        pib_values = trend + seasonal + noise
        
        # IVA correlacionado con PIB
        iva_values = (pib_values * 0.15 + np.random.normal(0, 500, len(fechas)))
        
        # Crear DataFrames
        df_pib = pd.DataFrame({
            'fecha': fechas,
            'valor': pib_values
        })
        
        df_iva = pd.DataFrame({
            'fecha': fechas,
            'valor': iva_values
        })
        
        # Guardar archivos
        df_pib.to_csv(self.ruta_datos / "pib_historico.csv", index=False)
        df_iva.to_csv(self.ruta_datos / "iva_historico.csv", index=False)
        
        return df_pib, df_iva
        
    def cargar_datos(self):
        """Cargar datos existentes o generar ejemplos"""
        ruta_pib = self.ruta_datos / "pib_historico.csv"
        ruta_iva = self.ruta_datos / "iva_historico.csv"
        
        if not (ruta_pib.exists() and ruta_iva.exists()):
            print("Archivos de datos no encontrados. Generando datos de ejemplo...")
            return self.generar_datos_ejemplo()
        
        try:
            df_pib = pd.read_csv(ruta_pib, parse_dates=['fecha'])
            df_iva = pd.read_csv(ruta_iva, parse_dates=['fecha'])
            print(f"Datos cargados: PIB ({len(df_pib)} registros), IVA ({len(df_iva)} registros)")
            return df_pib, df_iva
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return self.generar_datos_ejemplo()
    
    def ejecutar_analisis_completo(self):
        """Ejecutar análisis completo del proyecto"""
        print("=== INICIANDO ANÁLISIS PIB-IVA ===\n")
        
        # 1. Cargar datos
        df_pib, df_iva = self.cargar_datos()
        
        # 2. Preprocesamiento
        print("1. Preprocesando datos...")
        preprocesador = PreprocesadorPIBIVA()
        preprocesador.df_pib = df_pib
        preprocesador.df_iva = df_iva
        preprocesador.limpiar_datos()
        
        # 3. Crear ventanas temporales
        print("2. Creando ventanas temporales...")
        X, y = preprocesador.crear_ventanas_temporales()
        X_train, X_test, y_train, y_test = preprocesador.dividir_datos(X, y)
        
        print(f"   Datos de entrenamiento: {X_train.shape}")
        print(f"   Datos de prueba: {X_test.shape}")
        
        # 4. Modelo simple para demostración
        print("3. Entrenando modelo simple...")
        y_pred = self.modelo_simple(X_test, y_test)
        
        # 5. Evaluación
        print("4. Evaluando modelo...")
        evaluador = EvaluadorModelos()
        metricas = evaluador.calcular_metricas(y_test, y_pred, "Modelo Simple")
        
        for metrica, valor in metricas.items():
            print(f"   {metrica}: {valor:.4f}")
        
        # 6. Visualizaciones
        print("5. Generando visualizaciones...")
        visualizador = VisualizadorPIBIVA()
        
        # Combinar datos para visualizaciones
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        # Mostrar visualizaciones
        visualizador.serie_temporal_comparativa(df_pib, df_iva)
        visualizador.analisis_estacionalidad(df_pib, 'valor')
        evaluador.graficar_predicciones(y_test, y_pred)
        evaluador.analizar_residuos(y_test, y_pred)
        
        print("\n=== ANÁLISIS COMPLETADO ===")
        return metricas
    
    def modelo_simple(self, X_test, y_test):
        """Modelo simple para demostración"""
        # Media móvil simple como baseline
        y_pred = np.mean(X_test[:, :, 0], axis=1)  # Promedio de la ventana temporal
        return y_pred

if __name__ == "__main__":
    proyecto = ProyectoPIBIVA()
    proyecto.ejecutar_analisis_completo()
