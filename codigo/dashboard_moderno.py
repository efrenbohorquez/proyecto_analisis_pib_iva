import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DashboardIVA:
    """Dashboard moderno para análisis y predicción de IVA"""
    
    def __init__(self):
        # Configurar tema moderno
        self.color_pib = '#1f77b4'  # Azul
        self.color_iva = '#ff7f0e'  # Naranja
        self.color_pred = '#2ca02c'  # Verde
        self.color_residuos = '#d62728'  # Rojo
        
    def crear_dashboard_completo(self, df_pib, df_iva, y_test, y_pred, predicciones_futuras):
        """Crear dashboard completo con todas las visualizaciones"""
        print("🎨 Generando visualizaciones del dashboard...")
        
        # 1. Series temporales principales
        self.grafico_series_temporales(df_pib, df_iva)
        
        # 2. Análisis de correlación
        self.analisis_correlacion_avanzado(df_pib, df_iva)
        
        # 3. Evaluación del modelo
        self.visualizar_performance_modelo(y_test, y_pred)
        
        # 4. Predicciones futuras
        self.grafico_predicciones_futuras(df_iva, predicciones_futuras)
        
        # 5. Dashboard de métricas
        self.dashboard_metricas_interactivo(y_test, y_pred)
        
        # 6. Análisis de estacionalidad
        self.analisis_estacionalidad_moderno(df_iva)
        
        print("✅ Dashboard generado completamente")
    
    def grafico_series_temporales(self, df_pib, df_iva):
        """Gráfico moderno de series temporales con plotly"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📈 Evolución del PIB', '💰 Evolución del IVA'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # PIB
        fig.add_trace(
            go.Scatter(
                x=df_pib['fecha'],
                y=df_pib['valor'],
                name='PIB',
                line=dict(color=self.color_pib, width=3),
                hovertemplate='<b>PIB</b><br>Fecha: %{x}<br>Valor: %{y:.1f} mil millones €<extra></extra>'
            ),
            row=1, col=1
        )
        
        # IVA
        fig.add_trace(
            go.Scatter(
                x=df_iva['fecha'],
                y=df_iva['valor'],
                name='IVA',
                line=dict(color=self.color_iva, width=3),
                hovertemplate='<b>IVA</b><br>Fecha: %{x}<br>Valor: %{y:.1f} mil millones €<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Añadir medias móviles
        if len(df_pib) >= 12:
            df_pib_copy = df_pib.copy()
            df_pib_copy['ma12'] = df_pib_copy['valor'].rolling(12).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_pib_copy['fecha'],
                    y=df_pib_copy['ma12'],
                    name='PIB MA(12)',
                    line=dict(color=self.color_pib, width=2, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        if len(df_iva) >= 12:
            df_iva_copy = df_iva.copy()
            df_iva_copy['ma12'] = df_iva_copy['valor'].rolling(12).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_iva_copy['fecha'],
                    y=df_iva_copy['ma12'],
                    name='IVA MA(12)',
                    line=dict(color=self.color_iva, width=2, dash='dash'),
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='📊 Series Temporales PIB e IVA - España',
            height=600,
            showlegend=True,
            template='plotly_white',
            font=dict(size=12),
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="PIB (mil millones €)", row=1, col=1)
        fig.update_yaxes(title_text="IVA (mil millones €)", row=2, col=1)
        
        fig.show()
    
    def analisis_correlacion_avanzado(self, df_pib, df_iva):
        """Análisis de correlación avanzado con diferentes lags"""
        # Combinar datos
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        # Calcular correlaciones con diferentes lags
        lags = range(-6, 7)  # De -6 a +6 meses
        correlaciones = []
        
        for lag in lags:
            if lag < 0:
                corr = df_combined['valor_pib'].corr(df_combined['valor_iva'].shift(-lag))
            else:
                corr = df_combined['valor_pib'].shift(lag).corr(df_combined['valor_iva'])
            correlaciones.append(corr)
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('🔗 Correlación PIB-IVA por Lag', '📊 Scatter Plot PIB vs IVA'),
            horizontal_spacing=0.1
        )
        
        # Gráfico de correlaciones por lag
        colors = ['red' if corr < 0 else 'green' for corr in correlaciones]
        fig.add_trace(
            go.Bar(
                x=lags,
                y=correlaciones,
                marker_color=colors,
                name='Correlación',
                hovertemplate='Lag: %{x} meses<br>Correlación: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Scatter plot PIB vs IVA
        fig.add_trace(
            go.Scatter(
                x=df_combined['valor_pib'],
                y=df_combined['valor_iva'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.color_iva,
                    opacity=0.7
                ),
                name='PIB vs IVA',
                hovertemplate='PIB: %{x:.1f}<br>IVA: %{y:.1f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Línea de tendencia
        z = np.polyfit(df_combined['valor_pib'], df_combined['valor_iva'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df_combined['valor_pib'],
                y=p(df_combined['valor_pib']),
                mode='lines',
                line=dict(color='red', width=2),
                name='Tendencia',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='🔍 Análisis de Correlación PIB-IVA',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Lag (meses)", row=1, col=1)
        fig.update_xaxes(title_text="PIB (mil millones €)", row=1, col=2)
        fig.update_yaxes(title_text="Correlación", row=1, col=1)
        fig.update_yaxes(title_text="IVA (mil millones €)", row=1, col=2)
        
        fig.show()
        
        # Mostrar correlación máxima
        max_corr_idx = np.argmax(np.abs(correlaciones))
        max_corr_lag = lags[max_corr_idx]
        max_corr_value = correlaciones[max_corr_idx]
        print(f"📈 Correlación máxima: {max_corr_value:.3f} con lag de {max_corr_lag} meses")
    
    def visualizar_performance_modelo(self, y_test, y_pred):
        """Visualizar performance del modelo de predicción"""
        # Calcular residuos
        residuos = y_test - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🎯 Predicciones vs Valores Reales',
                '📊 Distribución de Residuos', 
                '📈 Residuos vs Predicciones',
                '⏱️ Residuos en el Tiempo'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Predicciones vs Reales
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(size=8, color=self.color_pred, opacity=0.7),
                name='Predicciones',
                hovertemplate='Real: %{x:.3f}<br>Predicción: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Línea diagonal perfecta
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Predicción Perfecta',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Histograma de residuos
        fig.add_trace(
            go.Histogram(
                x=residuos,
                nbinsx=20,
                marker_color=self.color_residuos,
                opacity=0.7,
                name='Residuos'
            ),
            row=1, col=2
        )
        
        # 3. Residuos vs Predicciones
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuos,
                mode='markers',
                marker=dict(size=6, color=self.color_residuos, opacity=0.7),
                name='Residuos vs Pred'
            ),
            row=2, col=1
        )
        
        # Línea en cero
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Residuos en el tiempo
        fig.add_trace(
            go.Scatter(
                x=list(range(len(residuos))),
                y=residuos,
                mode='lines+markers',
                line=dict(color=self.color_residuos),
                marker=dict(size=4),
                name='Residuos Temporales'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='🔍 Evaluación del Modelo de Predicción de IVA',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        fig.show()
    
    def grafico_predicciones_futuras(self, df_iva, predicciones_futuras):
        """Gráfico de predicciones futuras del IVA"""
        if predicciones_futuras is None or len(predicciones_futuras) == 0:
            return
        
        # Crear fechas futuras
        ultima_fecha = df_iva['fecha'].max()
        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.DateOffset(months=1),
            periods=len(predicciones_futuras),
            freq='M'
        )
        
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(
            go.Scatter(
                x=df_iva['fecha'],
                y=df_iva['valor'],
                name='IVA Histórico',
                line=dict(color=self.color_iva, width=3),
                hovertemplate='<b>Histórico</b><br>Fecha: %{x}<br>IVA: %{y:.1f} mil millones €<extra></extra>'
            )
        )
        
        # Predicciones futuras
        fig.add_trace(
            go.Scatter(
                x=fechas_futuras,
                y=predicciones_futuras,
                name='Predicciones',
                line=dict(color=self.color_pred, width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Predicción</b><br>Fecha: %{x}<br>IVA: %{y:.1f} mil millones €<extra></extra>'
            )
        )
        
        # Intervalo de confianza (simulado)
        margen_error = np.std(predicciones_futuras) * 0.5
        fig.add_trace(
            go.Scatter(
                x=fechas_futuras,
                y=predicciones_futuras + margen_error,
                mode='lines',
                line=dict(color=self.color_pred, width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=fechas_futuras,
                y=predicciones_futuras - margen_error,
                mode='lines',
                line=dict(color=self.color_pred, width=0),
                fill='tonexty',
                fillcolor=f'rgba(44, 160, 44, 0.2)',
                name='Intervalo Confianza',
                hoverinfo='skip'
            )
        )
        
        fig.update_layout(
            title='🔮 Predicciones Futuras del IVA',
            xaxis_title='Fecha',
            yaxis_title='IVA (mil millones €)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        # Añadir línea vertical separando histórico de predicciones
        fig.add_vline(
            x=ultima_fecha,
            line_dash="dash",
            line_color="gray",
            annotation_text="Inicio Predicciones"
        )
        
        fig.show()
    
    def dashboard_metricas_interactivo(self, y_test, y_pred):
        """Dashboard interactivo de métricas"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-8, y_test))) * 100
        r2 = r2_score(y_test, y_pred)
        
        # Crear gauge charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE', 'MAE', 'MAPE (%)', 'R²'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # RMSE
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=rmse,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RMSE"},
                gauge={
                    'axis': {'range': [None, 0.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.05], 'color': "lightgreen"},
                        {'range': [0.05, 0.1], 'color': "yellow"},
                        {'range': [0.1, 0.2], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.1
                    }
                }
            ),
            row=1, col=1
        )
        
        # MAE
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=mae,
                title={'text': "MAE"},
                gauge={
                    'axis': {'range': [None, 0.15]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.03], 'color': "lightgreen"},
                        {'range': [0.03, 0.08], 'color': "yellow"},
                        {'range': [0.08, 0.15], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # MAPE
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=mape,
                title={'text': "MAPE (%)"},
                gauge={
                    'axis': {'range': [None, 20]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "red"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # R²
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=r2,
                title={'text': "R²"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.7], 'color': "red"},
                        {'range': [0.7, 0.85], 'color': "yellow"},
                        {'range': [0.85, 1], 'color': "lightgreen"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='📊 Dashboard de Métricas del Modelo',
            height=600,
            template='plotly_white'
        )
        
        fig.show()
    
    def analisis_estacionalidad_moderno(self, df_iva):
        """Análisis de estacionalidad con visualizaciones modernas"""
        df = df_iva.copy()
        df['mes'] = df['fecha'].dt.month
        df['año'] = df['fecha'].dt.year
        df['trimestre'] = df['fecha'].dt.quarter
        
        # Crear datos mensuales y trimestrales
        monthly_avg = df.groupby('mes')['valor'].agg(['mean', 'std']).reset_index()
        quarterly_avg = df.groupby('trimestre')['valor'].agg(['mean', 'std']).reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📅 Patrón Estacional Mensual',
                '📊 Patrón Estacional Trimestral',
                '🌊 Descomposición de Tendencia',
                '📈 Evolución Anual'
            )
        )
        
        # 1. Patrón mensual
        fig.add_trace(
            go.Bar(
                x=monthly_avg['mes'],
                y=monthly_avg['mean'],
                error_y=dict(type='data', array=monthly_avg['std']),
                marker_color=self.color_iva,
                name='Promedio Mensual',
                hovertemplate='Mes: %{x}<br>IVA Promedio: %{y:.1f}<br>Desv. Std: %{error_y.array:.1f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Patrón trimestral
        fig.add_trace(
            go.Bar(
                x=['Q1', 'Q2', 'Q3', 'Q4'],
                y=quarterly_avg['mean'],
                error_y=dict(type='data', array=quarterly_avg['std']),
                marker_color=self.color_pred,
                name='Promedio Trimestral'
            ),
            row=1, col=2
        )
        
        # 3. Tendencia (media móvil)
        if len(df) >= 12:
            df['tendencia'] = df['valor'].rolling(12, center=True).mean()
            df['estacional'] = df['valor'] - df['tendencia']
            
            fig.add_trace(
                go.Scatter(
                    x=df['fecha'],
                    y=df['tendencia'],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='Tendencia',
                    hovertemplate='Fecha: %{x}<br>Tendencia: %{y:.1f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['fecha'],
                    y=df['estacional'],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Componente Estacional'
                ),
                row=2, col=1
            )
        
        # 4. Evolución anual
        yearly_avg = df.groupby('año')['valor'].mean().reset_index()
        if len(yearly_avg) > 1:
            fig.add_trace(
                go.Scatter(
                    x=yearly_avg['año'],
                    y=yearly_avg['valor'],
                    mode='lines+markers',
                    line=dict(color=self.color_iva, width=3),
                    marker=dict(size=10),
                    name='Promedio Anual',
                    hovertemplate='Año: %{x}<br>IVA Promedio: %{y:.1f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='🔄 Análisis de Estacionalidad del IVA',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        fig.show()
