import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DashboardCompletoIVA:
    """Dashboard completo con flujo analítico IVA → PIB → Modelos → Análisis → Recomendaciones"""
    
    def __init__(self):
        self.color_iva = '#ff7f0e'  # Naranja
        self.color_pib = '#1f77b4'  # Azul
        self.color_pred = '#2ca02c'  # Verde
        self.color_error = '#d62728'  # Rojo
        self.modelos_resultados = {}
        
    def ejecutar_dashboard_completo(self, df_pib, df_iva):
        """Ejecutar dashboard completo en orden lógico"""
        print("🚀 INICIANDO DASHBOARD ANALÍTICO COMPLETO")
        print("=" * 60)
        
        # 1. ANÁLISIS DE IVA
        print("\n📊 1. ANÁLISIS DE LA SERIE IVA")
        self.analizar_serie_iva(df_iva)
        
        # 2. PREPROCESADO DE IVA
        print("\n🔧 2. PREPROCESADO Y LIMPIEZA DE IVA")
        df_iva_limpio = self.preprocesar_serie_iva(df_iva)
        
        # 3. ANÁLISIS DE PIB
        print("\n📈 3. ANÁLISIS DE LA SERIE PIB")
        self.analizar_serie_pib(df_pib)
        
        # 4. PREPROCESADO DE PIB
        print("\n🔧 4. PREPROCESADO Y LIMPIEZA DE PIB")
        df_pib_limpio = self.preprocesar_serie_pib(df_pib)
        
        # 5. PREPARACIÓN DE DATOS PARA MODELOS
        print("\n🎯 5. PREPARACIÓN DE DATOS PARA MODELOS")
        X, y = self.preparar_datos_modelos(df_pib_limpio, df_iva_limpio)
        
        # 6. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
        print("\n🤖 6. ENTRENAMIENTO DE MÚLTIPLES MODELOS")
        self.entrenar_multiples_modelos(X, y)
        
        # 7. ANÁLISIS COMPARATIVO
        print("\n📊 7. ANÁLISIS COMPARATIVO DE MODELOS")
        self.analisis_comparativo_modelos()
        
        # 8. RECOMENDACIONES
        print("\n💡 8. RECOMENDACIONES")
        self.generar_recomendaciones()
        
        # 9. TOP 3 MODELOS MÁS ROBUSTOS
        print("\n🏆 9. TOP 3 MODELOS MÁS ROBUSTOS")
        self.mostrar_top3_modelos()
        
        print("\n✅ DASHBOARD COMPLETO FINALIZADO")
        
    def analizar_serie_iva(self, df_iva):
        """1. Análisis completo de la serie IVA"""
        # Estadísticas descriptivas
        stats = df_iva['valor'].describe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📈 Serie Temporal IVA',
                '📊 Distribución de Valores',
                '🔄 Autocorrelación',
                '📋 Estadísticas Descriptivas'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "histogram"}, {"type": "table"}]
            ]
        )
        
        # Serie temporal
        fig.add_trace(
            go.Scatter(
                x=df_iva['fecha'],
                y=df_iva['valor'],
                mode='lines',
                name='IVA',
                line=dict(color=self.color_iva, width=3),
                hovertemplate='Fecha: %{x}<br>IVA: %{y:.1f} mil millones €<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(
                x=df_iva['valor'],
                nbinsx=20,
                name='Distribución',
                marker_color=self.color_iva,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Tabla de estadísticas
        fig.add_trace(
            go.Table(
                header=dict(values=['Estadística', 'Valor'], fill_color='lightgray'),
                cells=dict(values=[
                    ['Media', 'Mediana', 'Desv. Std', 'Mín', 'Máx', 'CV%'],
                    [f'{stats["mean"]:.1f}', f'{stats["50%"]:.1f}', f'{stats["std"]:.1f}',
                     f'{stats["min"]:.1f}', f'{stats["max"]:.1f}', f'{(stats["std"]/stats["mean"])*100:.1f}']
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='📊 ANÁLISIS EXPLORATORIO DE LA SERIE IVA',
            height=600,
            template='plotly_white'
        )
        
        fig.show()
        
        # Mostrar insights
        cv = (stats['std'] / stats['mean']) * 100
        print(f"   💡 Coeficiente de Variación: {cv:.1f}% - {'Alta' if cv > 15 else 'Moderada' if cv > 10 else 'Baja'} volatilidad")
        print(f"   📈 Crecimiento total: {((df_iva['valor'].iloc[-1] / df_iva['valor'].iloc[0]) - 1) * 100:.1f}%")
        
    def preprocesar_serie_iva(self, df_iva):
        """2. Preprocesado de la serie IVA"""
        df_clean = df_iva.copy()
        
        # Detectar outliers
        Q1 = df_clean['valor'].quantile(0.25)
        Q3 = df_clean['valor'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_inf = df_clean['valor'] < (Q1 - 1.5 * IQR)
        outliers_sup = df_clean['valor'] > (Q3 + 1.5 * IQR)
        total_outliers = outliers_inf.sum() + outliers_sup.sum()
        
        # Crear visualización del preprocesado
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🔍 Detección de Outliers',
                '📊 Box Plot',
                '🧹 Serie Limpia vs Original',
                '📈 Tendencia y Estacionalidad'
            )
        )
        
        # Outliers
        fig.add_trace(
            go.Scatter(
                x=df_clean['fecha'],
                y=df_clean['valor'],
                mode='markers',
                marker=dict(
                    color=np.where(outliers_inf | outliers_sup, 'red', self.color_iva),
                    size=6
                ),
                name='IVA (outliers en rojo)',
                hovertemplate='Fecha: %{x}<br>Valor: %{y:.1f}<br>Outlier: %{marker.color}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=df_clean['valor'],
                name='IVA',
                marker_color=self.color_iva,
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        # Limpiar outliers (reemplazar por mediana)
        df_clean.loc[outliers_inf | outliers_sup, 'valor'] = df_clean['valor'].median()
        
        # Serie limpia vs original
        fig.add_trace(
            go.Scatter(
                x=df_iva['fecha'],
                y=df_iva['valor'],
                mode='lines',
                name='Original',
                line=dict(color='gray', width=2, dash='dot'),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_clean['fecha'],
                y=df_clean['valor'],
                mode='lines',
                name='Limpia',
                line=dict(color=self.color_iva, width=3)
            ),
            row=2, col=1
        )
        
        # Descomposición (tendencia)
        if len(df_clean) >= 12:
            df_clean['tendencia'] = df_clean['valor'].rolling(12, center=True).mean()
            df_clean['estacional'] = df_clean['valor'] - df_clean['tendencia']
            
            fig.add_trace(
                go.Scatter(
                    x=df_clean['fecha'],
                    y=df_clean['tendencia'],
                    mode='lines',
                    name='Tendencia',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_clean['fecha'],
                    y=df_clean['estacional'],
                    mode='lines',
                    name='Estacional',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='🔧 PREPROCESAMIENTO DE LA SERIE IVA',
            height=700,
            template='plotly_white'
        )
        
        fig.show()
        
        print(f"   🎯 Outliers detectados y corregidos: {total_outliers}")
        print(f"   ✅ Serie IVA preprocesada: {len(df_clean)} observaciones")
        
        return df_clean
        
    def analizar_serie_pib(self, df_pib):
        """3. Análisis completo de la serie PIB"""
        # ...existing code for PIB analysis similar to IVA...
        stats = df_pib['valor'].describe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📈 Serie Temporal PIB',
                '📊 Distribución de Valores',
                '🔗 Correlación con Lags',
                '📋 Estadísticas Descriptivas'
            )
        )
        
        # Serie temporal PIB
        fig.add_trace(
            go.Scatter(
                x=df_pib['fecha'],
                y=df_pib['valor'],
                mode='lines',
                name='PIB',
                line=dict(color=self.color_pib, width=3)
            ),
            row=1, col=1
        )
        
        # Distribución
        fig.add_trace(
            go.Histogram(
                x=df_pib['valor'],
                nbinsx=20,
                name='Distribución PIB',
                marker_color=self.color_pib,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Autocorrelación
        lags = range(1, 13)
        autocorr = [df_pib['valor'].autocorr(lag) for lag in lags]
        
        fig.add_trace(
            go.Bar(
                x=lags,
                y=autocorr,
                name='Autocorrelación',
                marker_color=self.color_pib
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='📈 ANÁLISIS EXPLORATORIO DE LA SERIE PIB',
            height=600,
            template='plotly_white'
        )
        
        fig.show()
        
    def preprocesar_serie_pib(self, df_pib):
        """4. Preprocesado de la serie PIB"""
        # Similar preprocessing as IVA
        df_clean = df_pib.copy()
        
        # Detectar y limpiar outliers
        Q1 = df_clean['valor'].quantile(0.25)
        Q3 = df_clean['valor'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df_clean['valor'] < (Q1 - 1.5 * IQR)) | (df_clean['valor'] > (Q3 + 1.5 * IQR))
        
        df_clean.loc[outliers, 'valor'] = df_clean['valor'].median()
        
        print(f"   🎯 PIB outliers corregidos: {outliers.sum()}")
        print(f"   ✅ Serie PIB preprocesada: {len(df_clean)} observaciones")
        
        return df_clean
        
    def preparar_datos_modelos(self, df_pib, df_iva):
        """5. Preparar datos para entrenamiento de modelos"""
        # Combinar datasets
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        # Crear features
        df_combined['pib_lag1'] = df_combined['valor_pib'].shift(1)
        df_combined['pib_lag2'] = df_combined['valor_pib'].shift(2)
        df_combined['iva_lag1'] = df_combined['valor_iva'].shift(1)
        df_combined['pib_ma3'] = df_combined['valor_pib'].rolling(3).mean()
        df_combined['mes'] = df_combined['fecha'].dt.month
        df_combined['trimestre'] = df_combined['fecha'].dt.quarter
        
        # Seleccionar features
        features = ['valor_pib', 'pib_lag1', 'pib_lag2', 'iva_lag1', 'pib_ma3', 'mes', 'trimestre']
        
        # Limpiar NaN
        df_model = df_combined[features + ['valor_iva']].dropna()
        
        X = df_model[features].values
        y = df_model['valor_iva'].values
        
        print(f"   📊 Features preparadas: {len(features)}")
        print(f"   🎯 Observaciones para entrenamiento: {len(X)}")
        
        return X, y
        
    def entrenar_multiples_modelos(self, X, y):
        """6. Entrenar múltiples modelos y comparar"""
        # Dividir datos
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Definir modelos
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Elastic Net': ElasticNet(alpha=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        }
        
        # Entrenar y evaluar cada modelo
        for nombre, modelo in modelos.items():
            try:
                print(f"   🤖 Entrenando {nombre}...")
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                # Calcular métricas
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                self.modelos_resultados[nombre] = {
                    'modelo': modelo,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'score_compuesto': (r2 * 0.4) + ((1 - rmse/np.std(y_test)) * 0.3) + ((1 - mape/100) * 0.3)
                }
                
                print(f"      ✅ {nombre}: R²={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f}")
                
            except Exception as e:
                print(f"      ❌ Error en {nombre}: {str(e)}")
                
    def analisis_comparativo_modelos(self):
        """7. Análisis comparativo de todos los modelos"""
        if not self.modelos_resultados:
            print("   ⚠️ No hay modelos entrenados para comparar")
            return
            
        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame({
            nombre: {
                'RMSE': datos['rmse'],
                'MAE': datos['mae'],
                'R²': datos['r2'],
                'MAPE': datos['mape'],
                'Score': datos['score_compuesto']
            }
            for nombre, datos in self.modelos_resultados.items()
        }).T
        
        # Visualización comparativa
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 RMSE por Modelo', '📈 R² por Modelo', '🎯 MAE por Modelo', '📋 Ranking General')
        )
        
        modelos = list(df_resultados.index)
        colors = px.colors.qualitative.Set3[:len(modelos)]
        
        # RMSE
        fig.add_trace(
            go.Bar(x=modelos, y=df_resultados['RMSE'], marker_color=colors, name='RMSE'),
            row=1, col=1
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=modelos, y=df_resultados['R²'], marker_color=colors, name='R²'),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=modelos, y=df_resultados['MAE'], marker_color=colors, name='MAE'),
            row=2, col=1
        )
        
        # Ranking
        df_sorted = df_resultados.sort_values('Score', ascending=False)
        fig.add_trace(
            go.Bar(
                x=df_sorted.index,
                y=df_sorted['Score'],
                marker_color=['gold', 'silver', '#CD7F32'] + ['lightblue'] * (len(df_sorted) - 3),
                name='Score Compuesto'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='🏆 COMPARACIÓN DE MODELOS DE PREDICCIÓN DE IVA',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        fig.show()
        
        # Mostrar tabla de resultados
        print("\n📋 TABLA COMPARATIVA DE MODELOS:")
        print(df_resultados.round(3).to_string())
        
    def generar_recomendaciones(self):
        """8. Generar recomendaciones basadas en análisis"""
        if not self.modelos_resultados:
            return
            
        print("\n💡 RECOMENDACIONES BASADAS EN EL ANÁLISIS:")
        print("=" * 50)
        
        # Encontrar mejor modelo por métrica
        mejor_r2 = max(self.modelos_resultados.items(), key=lambda x: x[1]['r2'])
        mejor_rmse = min(self.modelos_resultados.items(), key=lambda x: x[1]['rmse'])
        mejor_mae = min(self.modelos_resultados.items(), key=lambda x: x[1]['mae'])
        
        print(f"🎯 MEJOR PRECISIÓN (R²): {mejor_r2[0]} ({mejor_r2[1]['r2']:.3f})")
        print(f"🎯 MENOR ERROR CUADRÁTICO: {mejor_rmse[0]} ({mejor_rmse[1]['rmse']:.1f})")
        print(f"🎯 MENOR ERROR ABSOLUTO: {mejor_mae[0]} ({mejor_mae[1]['mae']:.1f})")
        
        print("\n📊 RECOMENDACIONES ESTRATÉGICAS:")
        print("   1. Para PRODUCCIÓN: Usar ensemble de los 3 mejores modelos")
        print("   2. Para INTERPRETABILIDAD: Random Forest o Gradient Boosting")
        print("   3. Para RAPIDEZ: Ridge o Elastic Net")
        print("   4. Para PRECISIÓN MÁXIMA: Neural Network con tuning")
        
        print("\n🔧 MEJORAS SUGERIDAS:")
        print("   • Incluir más variables exógenas (tipos de interés, inflación)")
        print("   • Ampliar ventana temporal de datos")
        print("   • Implementar validación cruzada temporal")
        print("   • Optimizar hiperparámetros con Grid Search")
        
    def mostrar_top3_modelos(self):
        """9. Mostrar los 3 modelos más robustos"""
        if not self.modelos_resultados:
            return
            
        # Ordenar por score compuesto
        top3 = sorted(
            self.modelos_resultados.items(),
            key=lambda x: x[1]['score_compuesto'],
            reverse=True
        )[:3]
        
        print("\n🏆 TOP 3 MODELOS MÁS ROBUSTOS:")
        print("=" * 50)
        
        medallas = ['🥇', '🥈', '🥉']
        
        for i, (nombre, datos) in enumerate(top3):
            print(f"\n{medallas[i]} PUESTO {i+1}: {nombre}")
            print(f"   📊 Score Compuesto: {datos['score_compuesto']:.4f}")
            print(f"   📈 R²: {datos['r2']:.4f}")
            print(f"   📉 RMSE: {datos['rmse']:.2f}")
            print(f"   🎯 MAE: {datos['mae']:.2f}")
            print(f"   📋 MAPE: {datos['mape']:.2f}%")
            
            # Fortalezas específicas
            if i == 0:
                print("   💪 FORTALEZAS: Mejor balance general de métricas")
            elif i == 1:
                print("   💪 FORTALEZAS: Excelente como modelo de respaldo")
            else:
                print("   💪 FORTALEZAS: Buena opción para ensemble")
        
        # Visualización de TOP 3
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f'{medallas[i]} {nombre}' for i, (nombre, _) in enumerate(top3)],
            specs=[[{"type": "indicator"}] * 3]
        )
        
        for i, (nombre, datos) in enumerate(top3):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=datos['r2'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"R² - {nombre}"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': ['darkred', 'orange', 'gold'][i]},
                        'steps': [
                            {'range': [0, 0.7], 'color': "lightgray"},
                            {'range': [0.7, 0.85], 'color': "yellow"},
                            {'range': [0.85, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='🏆 TOP 3 MODELOS - INDICADORES R²',
            height=400,
            template='plotly_white'
        )
        
        fig.show()
        
        print(f"\n🎯 RECOMENDACIÓN FINAL:")
        print(f"   Implementar {top3[0][0]} como modelo principal")
        print(f"   Usar {top3[1][0]} como modelo de validación")
        print(f"   Ensemble de los 3 para máxima robustez")
