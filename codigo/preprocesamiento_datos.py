import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PreprocesadorPIBIVA:
    """Clase optimizada para preprocesamiento de datos PIB-IVA"""
    
    def __init__(self, ventana_temporal=12):
        self.scaler_pib = MinMaxScaler()
        self.scaler_iva = MinMaxScaler()
        self.ventana_temporal = ventana_temporal
        self.df_pib = None
        self.df_iva = None
        
    def cargar_datos(self, ruta_pib, ruta_iva):
        """Cargar y validar datos PIB e IVA"""
        try:
            self.df_pib = pd.read_csv(ruta_pib, parse_dates=['fecha'])
            self.df_iva = pd.read_csv(ruta_iva, parse_dates=['fecha'])
            return self._validar_datos()
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def _validar_datos(self):
        """Validar consistencia de datos"""
        if self.df_pib.empty or self.df_iva.empty:
            return False
        
        # Verificar columnas requeridas
        cols_requeridas = ['fecha', 'valor']
        for df in [self.df_pib, self.df_iva]:
            if not all(col in df.columns for col in cols_requeridas):
                return False
        return True
    
    def limpiar_datos(self):
        """Limpieza optimizada de datos"""
        if self.df_pib is None or self.df_iva is None:
            raise ValueError("Debe cargar los datos primero")
            
        # Eliminar duplicados y ordenar
        for df in [self.df_pib, self.df_iva]:
            df.drop_duplicates('fecha', inplace=True)
            df.sort_values('fecha', inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        # Tratar outliers y valores faltantes
        self._tratar_outliers()
        self._imputar_valores_faltantes()
        
        print(f"Datos limpiados: PIB {len(self.df_pib)} registros, IVA {len(self.df_iva)} registros")
    
    def _tratar_outliers(self, umbral_z=3):
        """Detectar y tratar outliers usando Z-score"""
        for df, nombre in [(self.df_pib, 'PIB'), (self.df_iva, 'IVA')]:
            z_scores = np.abs(stats.zscore(df['valor']))
            outliers = z_scores > umbral_z
            if outliers.sum() > 0:
                df.loc[outliers, 'valor'] = df['valor'].median()
                print(f"   Outliers corregidos en {nombre}: {outliers.sum()}")
    
    def _imputar_valores_faltantes(self):
        """Imputar valores faltantes con interpolación"""
        for df, nombre in [(self.df_pib, 'PIB'), (self.df_iva, 'IVA')]:
            nulos_antes = df['valor'].isnull().sum()
            if nulos_antes > 0:
                df['valor'] = df['valor'].interpolate(method='linear')
                print(f"   Valores imputados en {nombre}: {nulos_antes}")
    
    def crear_ventanas_temporales(self, target='pib'):
        """Crear secuencias temporales optimizadas"""
        # Combinar datasets por fecha
        df_combined = pd.merge(self.df_pib, self.df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        if len(df_combined) < self.ventana_temporal:
            raise ValueError(f"Necesita al menos {self.ventana_temporal} registros")
        
        # Normalizar datos
        pib_norm = self.scaler_pib.fit_transform(df_combined[['valor_pib']])
        iva_norm = self.scaler_iva.fit_transform(df_combined[['valor_iva']])
        
        # Crear ventanas deslizantes
        X, y = [], []
        for i in range(self.ventana_temporal, len(df_combined)):
            # Ventana de entrada (PIB e IVA)
            ventana = np.column_stack([
                pib_norm[i-self.ventana_temporal:i, 0],
                iva_norm[i-self.ventana_temporal:i, 0]
            ])
            X.append(ventana)
            
            # Target (siguiente valor de PIB o IVA)
            if target == 'pib':
                y.append(pib_norm[i, 0])
            else:
                y.append(iva_norm[i, 0])
        
        return np.array(X), np.array(y)
    
    def dividir_datos(self, X, y, test_size=0.2):
        """Dividir datos manteniendo orden temporal"""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def obtener_datos_originales(self):
        """Retornar datos originales para visualizaciones"""
        return self.df_pib.copy(), self.df_iva.copy()
