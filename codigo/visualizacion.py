import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class VisualizadorPIBIVA:
    """Clase simplificada para visualizaciones del proyecto PIB-IVA"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
    def serie_temporal_comparativa(self, df_pib, df_iva):
        """Gráfico comparativo optimizado de series temporales"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # PIB
        axes[0].plot(df_pib['fecha'], df_pib['valor'], linewidth=2, color='blue', alpha=0.8)
        axes[0].set_title('Evolución del PIB', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('PIB (millones)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # IVA
        axes[1].plot(df_iva['fecha'], df_iva['valor'], linewidth=2, color='red', alpha=0.8)
        axes[1].set_title('Evolución del IVA', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('IVA (millones)', fontsize=12)
        axes[1].set_xlabel('Fecha', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar estadísticas básicas
        print("Estadísticas PIB:")
        print(f"  Promedio: {df_pib['valor'].mean():.2f}")
        print(f"  Desviación: {df_pib['valor'].std():.2f}")
        print(f"  Min: {df_pib['valor'].min():.2f}")
        print(f"  Max: {df_pib['valor'].max():.2f}")
        
        print("\nEstadísticas IVA:")
        print(f"  Promedio: {df_iva['valor'].mean():.2f}")
        print(f"  Desviación: {df_iva['valor'].std():.2f}")
        print(f"  Min: {df_iva['valor'].min():.2f}")
        print(f"  Max: {df_iva['valor'].max():.2f}")
    
    def analisis_estacionalidad(self, df, columna_valor='valor'):
        """Análisis de estacionalidad simplificado"""
        # Preparar datos
        df = df.copy()
        df['mes'] = df['fecha'].dt.month
        df['año'] = df['fecha'].dt.year
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribución por mes
        monthly_data = df.groupby('mes')[columna_valor].mean()
        axes[0, 0].bar(monthly_data.index, monthly_data.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Promedio por Mes')
        axes[0, 0].set_xlabel('Mes')
        axes[0, 0].set_ylabel('Valor Promedio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tendencia anual
        yearly_data = df.groupby('año')[columna_valor].mean()
        axes[0, 1].plot(yearly_data.index, yearly_data.values, marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Tendencia Anual')
        axes[0, 1].set_xlabel('Año')
        axes[0, 1].set_ylabel('Valor Promedio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Serie original con media móvil
        axes[1, 0].plot(df['fecha'], df[columna_valor], alpha=0.6, label='Original', linewidth=1)
        if len(df) >= 12:
            df['media_movil'] = df[columna_valor].rolling(window=12, center=True).mean()
            axes[1, 0].plot(df['fecha'], df['media_movil'], linewidth=2, label='Media Móvil (12m)', color='red')
        axes[1, 0].set_title('Serie Original vs Media Móvil')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Variabilidad por mes
        monthly_std = df.groupby('mes')[columna_valor].std()
        axes[1, 1].bar(monthly_std.index, monthly_std.values, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Variabilidad por Mes')
        axes[1, 1].set_xlabel('Mes')
        axes[1, 1].set_ylabel('Desviación Estándar')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def correlacion_pib_iva(self, df_pib, df_iva):
        """Análisis de correlación entre PIB e IVA"""
        # Combinar datos
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        if len(df_combined) < 2:
            print("Datos insuficientes para análisis de correlación")
            return
        
        # Calcular correlación
        correlacion = df_combined['valor_pib'].corr(df_combined['valor_iva'])
        
        # Visualizar
        plt.figure(figsize=(10, 6))
        plt.scatter(df_combined['valor_pib'], df_combined['valor_iva'], alpha=0.7, s=50)
        
        # Línea de tendencia
        z = np.polyfit(df_combined['valor_pib'], df_combined['valor_iva'], 1)
        p = np.poly1d(z)
        plt.plot(df_combined['valor_pib'], p(df_combined['valor_pib']), "r--", linewidth=2)
        
        plt.xlabel('PIB')
        plt.ylabel('IVA')
        plt.title(f'Correlación PIB-IVA (r = {correlacion:.3f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Correlación PIB-IVA: {correlacion:.4f}")
        
        return correlacion
