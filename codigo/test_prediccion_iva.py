import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor_iva import PredictorIVA
from dashboard_moderno import DashboardIVA

class TestPredictorIVA:
    """Script de prueba para predicción de IVA usando PIB como variable exógena"""
    
    def __init__(self):
        self.ruta_base = Path(__file__).parent.parent
        self.ruta_datos = self.ruta_base / "datos"
        self.predictor = PredictorIVA()
        self.dashboard = DashboardIVA()
        
    def generar_datos_reales(self):
        """Generar datos realistas basados en economía española"""
        print("📊 Generando datos económicos realistas...")
        
        # Crear 60 meses de datos (2019-2024)
        fechas = pd.date_range('2019-01-01', '2023-12-31', freq='M')
        
        # PIB España (valores en miles de millones €, tendencia real)
        base_pib = 1200  # PIB base España
        trend_pib = np.linspace(0, 50, len(fechas))  # Crecimiento gradual
        seasonal_pib = 20 * np.sin(2 * np.pi * np.arange(len(fechas)) / 12)  # Estacionalidad
        covid_impact = np.where((np.arange(len(fechas)) >= 15) & (np.arange(len(fechas)) <= 18), -80, 0)  # Impacto COVID
        noise_pib = np.random.normal(0, 8, len(fechas))
        pib_values = base_pib + trend_pib + seasonal_pib + covid_impact + noise_pib
        
        # IVA fuertemente correlacionado con PIB (tasa efectiva ~11-12% del PIB)
        base_iva_rate = 0.115  # Tasa base IVA/PIB
        seasonal_iva = 0.01 * np.sin(2 * np.pi * np.arange(len(fechas)) / 12 + np.pi/3)  # Estacionalidad diferente
        economic_cycles = 0.005 * np.sin(2 * np.pi * np.arange(len(fechas)) / 24)  # Ciclos económicos
        noise_iva = np.random.normal(0, 0.003, len(fechas))
        iva_rate = base_iva_rate + seasonal_iva + economic_cycles + noise_iva
        iva_values = pib_values * iva_rate
        
        # Crear DataFrames
        df_pib = pd.DataFrame({
            'fecha': fechas,
            'valor': pib_values,
            'tipo': 'PIB'
        })
        
        df_iva = pd.DataFrame({
            'fecha': fechas,
            'valor': iva_values,
            'tipo': 'IVA'
        })
        
        # Guardar datos
        self.ruta_datos.mkdir(exist_ok=True)
        df_pib.to_csv(self.ruta_datos / "pib_historico.csv", index=False)
        df_iva.to_csv(self.ruta_datos / "iva_historico.csv", index=False)
        
        print(f"✅ Datos generados: {len(fechas)} observaciones mensuales")
        print(f"   PIB promedio: {pib_values.mean():.1f} mil millones €")
        print(f"   IVA promedio: {iva_values.mean():.1f} mil millones €")
        print(f"   Correlación PIB-IVA: {np.corrcoef(pib_values, iva_values)[0,1]:.3f}")
        
        return df_pib, df_iva
        
    def ejecutar_test_completo(self):
        """Ejecutar test completo de predicción de IVA"""
        print("🚀 INICIANDO TEST DE PREDICCIÓN DE IVA")
        print("=" * 50)
        
        # 1. Generar/cargar datos
        df_pib, df_iva = self.generar_datos_reales()
        
        # 2. Entrenar predictor
        print("\n🔧 Entrenando modelo de predicción de IVA...")
        metricas = self.predictor.entrenar_modelo(df_pib, df_iva)
        
        # 3. Realizar predicciones
        print("\n📈 Generando predicciones...")
        predicciones = self.predictor.predecir_iva(horizonte=6)
        
        # 4. Mostrar resultados
        print("\n📊 RESULTADOS DEL MODELO:")
        for metrica, valor in metricas.items():
            print(f"   {metrica}: {valor:.4f}")
        
        # 5. Generar dashboard
        print("\n🎨 Generando dashboard interactivo...")
        self.dashboard.crear_dashboard_completo(
            df_pib, df_iva, 
            self.predictor.y_test, 
            self.predictor.y_pred,
            predicciones
        )
        
        # 6. Análisis económico
        self.analisis_economico(df_pib, df_iva, predicciones)
        
        print("\n✅ TEST COMPLETADO EXITOSAMENTE")
        return metricas, predicciones
    
    def analisis_economico(self, df_pib, df_iva, predicciones):
        """Análisis económico de los resultados"""
        print("\n📋 ANÁLISIS ECONÓMICO:")
        
        # Combinar datos
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        # Calcular ratio IVA/PIB
        ratio_iva_pib = df_combined['valor_iva'].mean() / df_combined['valor_pib'].mean()
        print(f"   Ratio promedio IVA/PIB: {ratio_iva_pib:.1%}")
        
        # Elasticidad IVA respecto a PIB
        correlacion = df_combined['valor_pib'].corr(df_combined['valor_iva'])
        print(f"   Correlación PIB-IVA: {correlacion:.3f}")
        
        # Volatilidad
        vol_pib = df_combined['valor_pib'].std() / df_combined['valor_pib'].mean()
        vol_iva = df_combined['valor_iva'].std() / df_combined['valor_iva'].mean()
        print(f"   Volatilidad PIB: {vol_pib:.1%}")
        print(f"   Volatilidad IVA: {vol_iva:.1%}")
        
        # Proyección del ratio
        if predicciones is not None and len(predicciones) > 0:
            ultimo_pib = df_combined['valor_pib'].iloc[-1]
            pred_iva_promedio = np.mean(predicciones)
            ratio_proyectado = pred_iva_promedio / ultimo_pib
            print(f"   Ratio proyectado IVA/PIB: {ratio_proyectado:.1%}")

if __name__ == "__main__":
    test = TestPredictorIVA()
    test.ejecutar_test_completo()
