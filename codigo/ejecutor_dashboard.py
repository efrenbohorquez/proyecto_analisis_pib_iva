import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard_completo import DashboardCompletoIVA

class EjecutorDashboard:
    """Ejecutor principal del dashboard completo de análisis IVA-PIB"""
    
    def __init__(self):
        self.ruta_base = Path(__file__).parent.parent
        self.ruta_datos = self.ruta_base / "datos"
        self.dashboard = DashboardCompletoIVA()
        
    def generar_datos_economicos(self):
        """Generar datos económicos realistas para España"""
        print("📊 Generando datos económicos para el análisis...")
        
        # 60 meses de datos (2019-2024)
        fechas = pd.date_range('2019-01-01', '2023-12-31', freq='M')
        
        # PIB España (valores realistas en miles de millones €)
        base_pib = 1250
        trend_pib = np.linspace(0, 80, len(fechas))
        seasonal_pib = 25 * np.sin(2 * np.pi * np.arange(len(fechas)) / 12)
        covid_impact = np.where((np.arange(len(fechas)) >= 15) & (np.arange(len(fechas)) <= 18), -100, 0)
        economic_cycles = 15 * np.sin(2 * np.pi * np.arange(len(fechas)) / 36)
        noise_pib = np.random.normal(0, 12, len(fechas))
        pib_values = base_pib + trend_pib + seasonal_pib + covid_impact + economic_cycles + noise_pib
        
        # IVA fuertemente correlacionado con PIB
        base_rate = 0.118  # Tasa base IVA/PIB
        seasonal_iva = 0.012 * np.sin(2 * np.pi * np.arange(len(fechas)) / 12 + np.pi/4)
        business_cycles = 0.008 * np.sin(2 * np.pi * np.arange(len(fechas)) / 30)
        policy_changes = np.where(np.arange(len(fechas)) % 24 == 0, np.random.normal(0, 0.005), 0)
        noise_iva = np.random.normal(0, 0.004, len(fechas))
        
        iva_rate = base_rate + seasonal_iva + business_cycles + policy_changes + noise_iva
        iva_values = pib_values * iva_rate
        
        # Añadir algunos outliers realistas
        outlier_indices = np.random.choice(len(fechas), 3, replace=False)
        pib_values[outlier_indices] *= np.random.uniform(0.95, 1.05, 3)
        iva_values[outlier_indices] *= np.random.uniform(0.92, 1.08, 3)
        
        # Crear DataFrames
        df_pib = pd.DataFrame({
            'fecha': fechas,
            'valor': pib_values
        })
        
        df_iva = pd.DataFrame({
            'fecha': fechas,
            'valor': iva_values
        })
        
        # Guardar datos
        self.ruta_datos.mkdir(exist_ok=True)
        df_pib.to_csv(self.ruta_datos / "pib_historico.csv", index=False)
        df_iva.to_csv(self.ruta_datos / "iva_historico.csv", index=False)
        
        print(f"✅ Datos generados y guardados:")
        print(f"   📈 PIB: {len(fechas)} observaciones, promedio {pib_values.mean():.1f} mil millones €")
        print(f"   💰 IVA: {len(fechas)} observaciones, promedio {iva_values.mean():.1f} mil millones €")
        print(f"   🔗 Correlación PIB-IVA: {np.corrcoef(pib_values, iva_values)[0,1]:.3f}")
        
        return df_pib, df_iva
        
    def ejecutar_analisis_completo(self):
        """Ejecutar análisis completo con dashboard"""
        print("🚀 EJECUTANDO ANÁLISIS COMPLETO PIB-IVA")
        print("=" * 60)
        
        # Generar datos
        df_pib, df_iva = self.generar_datos_economicos()
        
        # Ejecutar dashboard completo
        self.dashboard.ejecutar_dashboard_completo(df_pib, df_iva)
        
        print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("📊 Todas las visualizaciones han sido generadas")
        print("💡 Revise las recomendaciones para implementación")

if __name__ == "__main__":
    ejecutor = EjecutorDashboard()
    ejecutor.ejecutar_analisis_completo()
