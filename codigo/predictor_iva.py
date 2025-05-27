import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PredictorIVA:
    """Modelo especializado para predicción de IVA usando PIB como variable exógena"""
    
    def __init__(self, ventana_temporal=12):
        self.ventana_temporal = ventana_temporal
        self.scaler_pib = MinMaxScaler()
        self.scaler_iva = MinMaxScaler()
        self.modelo_principal = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.modelo_backup = ElasticNet(alpha=0.1, random_state=42)
        self.features_names = []
        self.is_trained = False
        
    def crear_features_exogenas(self, df_pib, df_iva):
        """Crear features avanzadas usando PIB como variable exógena"""
        # Combinar datasets
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        df_combined = df_combined.sort_values('fecha').reset_index(drop=True)
        
        # Features básicas normalizadas
        df_combined['pib_norm'] = self.scaler_pib.fit_transform(df_combined[['valor_pib']])
        df_combined['iva_norm'] = self.scaler_iva.fit_transform(df_combined[['valor_iva']])
        
        # Features derivadas del PIB (variable exógena)
        df_combined['pib_lag1'] = df_combined['pib_norm'].shift(1)
        df_combined['pib_lag2'] = df_combined['pib_norm'].shift(2)
        df_combined['pib_lag3'] = df_combined['pib_norm'].shift(3)
        df_combined['pib_ma3'] = df_combined['pib_norm'].rolling(3).mean()
        df_combined['pib_ma6'] = df_combined['pib_norm'].rolling(6).mean()
        df_combined['pib_std3'] = df_combined['pib_norm'].rolling(3).std()
        df_combined['pib_trend'] = df_combined['pib_norm'].diff()
        df_combined['pib_accel'] = df_combined['pib_trend'].diff()
        
        # Features temporales
        df_combined['mes'] = df_combined['fecha'].dt.month
        df_combined['trimestre'] = df_combined['fecha'].dt.quarter
        df_combined['mes_sin'] = np.sin(2 * np.pi * df_combined['mes'] / 12)
        df_combined['mes_cos'] = np.cos(2 * np.pi * df_combined['mes'] / 12)
        
        # Features de interacción PIB-temporal
        df_combined['pib_x_mes'] = df_combined['pib_norm'] * df_combined['mes_sin']
        df_combined['pib_x_trimestre'] = df_combined['pib_norm'] * df_combined['trimestre']
        
        # Features históricos del IVA (autorregresivos)
        df_combined['iva_lag1'] = df_combined['iva_norm'].shift(1)
        df_combined['iva_lag2'] = df_combined['iva_norm'].shift(2)
        df_combined['iva_ma3'] = df_combined['iva_norm'].rolling(3).mean()
        
        # Ratio PIB/IVA histórico
        df_combined['ratio_pib_iva'] = df_combined['pib_norm'] / (df_combined['iva_norm'] + 1e-8)
        df_combined['ratio_ma3'] = df_combined['ratio_pib_iva'].rolling(3).mean()
        
        return df_combined
    
    def preparar_datos_entrenamiento(self, df_combined):
        """Preparar matrices X, y para entrenamiento"""
        # Seleccionar features
        feature_cols = [
            'pib_norm', 'pib_lag1', 'pib_lag2', 'pib_lag3',
            'pib_ma3', 'pib_ma6', 'pib_std3', 'pib_trend', 'pib_accel',
            'mes_sin', 'mes_cos', 'trimestre',
            'pib_x_mes', 'pib_x_trimestre',
            'iva_lag1', 'iva_lag2', 'iva_ma3',
            'ratio_pib_iva', 'ratio_ma3'
        ]
        
        # Filtrar columnas que existen
        available_features = [col for col in feature_cols if col in df_combined.columns]
        self.features_names = available_features
        
        # Crear dataset sin NaN
        df_clean = df_combined[available_features + ['iva_norm']].dropna()
        
        if len(df_clean) < self.ventana_temporal:
            raise ValueError(f"Datos insuficientes. Se requieren al menos {self.ventana_temporal} observaciones")
        
        X = df_clean[available_features].values
        y = df_clean['iva_norm'].values
        
        return X, y
    
    def entrenar_modelo(self, df_pib, df_iva, test_size=0.2):
        """Entrenar modelo de predicción de IVA"""
        print("🔧 Preparando features exógenas...")
        df_combined = self.crear_features_exogenas(df_pib, df_iva)
        
        print("📊 Creando dataset de entrenamiento...")
        X, y = self.preparar_datos_entrenamiento(df_combined)
        
        # División temporal
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   Entrenamiento: {X_train.shape[0]} observaciones")
        print(f"   Prueba: {X_test.shape[0]} observaciones")
        print(f"   Features: {len(self.features_names)}")
        
        # Entrenar modelo principal
        print("🤖 Entrenando Random Forest...")
        self.modelo_principal.fit(X_train, y_train)
        
        # Entrenar modelo backup
        print("🤖 Entrenando modelo backup...")
        self.modelo_backup.fit(X_train, y_train)
        
        # Evaluar modelos
        y_pred_rf = self.modelo_principal.predict(X_test)
        y_pred_en = self.modelo_backup.predict(X_test)
        
        # Seleccionar mejor modelo
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        rmse_en = np.sqrt(mean_squared_error(y_test, y_pred_en))
        
        if rmse_rf <= rmse_en:
            self.modelo_activo = self.modelo_principal
            y_pred = y_pred_rf
            modelo_usado = "Random Forest"
        else:
            self.modelo_activo = self.modelo_backup
            y_pred = y_pred_en
            modelo_usado = "Elastic Net"
        
        # Guardar para visualizaciones
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.df_combined = df_combined
        
        # Calcular métricas
        metricas = self.calcular_metricas_completas(y_test, y_pred)
        metricas['Modelo_Usado'] = modelo_usado
        
        self.is_trained = True
        print(f"✅ Modelo entrenado: {modelo_usado}")
        
        return metricas
    
    def calcular_metricas_completas(self, y_true, y_pred):
        """Calcular métricas específicas para predicción de IVA"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Métricas específicas para IVA
        # Convertir de vuelta a escala original para interpretación
        y_true_orig = self.scaler_iva.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = self.scaler_iva.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        error_absoluto_millones = np.mean(np.abs(y_true_orig - y_pred_orig))
        error_relativo_promedio = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Error_Absoluto_Millones_€': error_absoluto_millones,
            'Error_Relativo_%': error_relativo_promedio
        }
    
    def predecir_iva(self, horizonte=6):
        """Predecir IVA para los próximos períodos"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecute entrenar_modelo() primero.")
        
        print(f"🔮 Prediciendo IVA para los próximos {horizonte} meses...")
        
        # Usar últimas observaciones como base
        last_features = self.X_test[-1:].copy()
        predicciones = []
        
        for i in range(horizonte):
            # Predecir próximo valor
            pred_normalizado = self.modelo_activo.predict(last_features)[0]
            pred_original = self.scaler_iva.inverse_transform([[pred_normalizado]])[0, 0]
            predicciones.append(pred_original)
            
            # Actualizar features para siguiente predicción (simplificado)
            # En un modelo real, necesitaríamos proyectar también el PIB
            last_features = last_features.copy()
            if len(self.features_names) > 0:
                # Simulación simple: mantener PIB y actualizar lags de IVA
                iva_idx = [i for i, name in enumerate(self.features_names) if 'iva_lag1' in name]
                if iva_idx:
                    last_features[0, iva_idx[0]] = pred_normalizado
        
        print(f"✅ Predicciones generadas: {len(predicciones)} valores")
        return np.array(predicciones)
    
    def obtener_importancia_features(self):
        """Obtener importancia de features del modelo"""
        if not self.is_trained or not hasattr(self.modelo_activo, 'feature_importances_'):
            return None
        
        importancia = self.modelo_activo.feature_importances_
        return dict(zip(self.features_names, importancia))
