import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

class EvaluadorModelos:
    """Clase simplificada para evaluación de modelos"""
    
    def __init__(self):
        self.metricas = {}
        plt.style.use('default')
    
    def calcular_metricas(self, y_true, y_pred, nombre_modelo="Modelo"):
        """Calcular métricas esenciales de evaluación"""
        # Validar entradas
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Métricas principales
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE con manejo de divisiones por cero
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
        
        # Correlación
        correlacion = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.metricas[nombre_modelo] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Correlación': correlacion,
            'R²': r2
        }
        
        return self.metricas[nombre_modelo]
    
    def graficar_predicciones(self, y_true, y_pred, titulo="Predicciones vs Valores Reales"):
        """Visualizar predicciones vs valores reales"""
        plt.figure(figsize=(12, 6))
        
        indices = range(len(y_true))
        plt.plot(indices, y_true, label='Valores Reales', linewidth=2, alpha=0.8)
        plt.plot(indices, y_pred, label='Predicciones', linewidth=2, alpha=0.8)
        
        plt.xlabel('Índice Temporal')
        plt.ylabel('Valor Normalizado')
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analizar_residuos(self, y_true, y_pred, titulo="Análisis de Residuos"):
        """Análisis simplificado de residuos"""
        residuos = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuos vs predicciones
        axes[0].scatter(y_pred, residuos, alpha=0.6, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicciones')
        axes[0].set_ylabel('Residuos')
        axes[0].set_title('Residuos vs Predicciones')
        axes[0].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[1].hist(residuos, bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuos')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Residuos')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(titulo, fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Estadísticas de residuos
        print(f"Estadísticas de residuos:")
        print(f"  Media: {np.mean(residuos):.4f}")
        print(f"  Desviación estándar: {np.std(residuos):.4f}")
        print(f"  Min: {np.min(residuos):.4f}")
        print(f"  Max: {np.max(residuos):.4f}")
