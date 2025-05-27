"""
Visualizaciones de calidad académica para análisis PIB-IVA
Diseñadas específicamente para presentaciones de maestría en ciencia de datos
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.config import config
from src.utils.logger import get_logger

class AcademicPlotGenerator:
    """Generador de visualizaciones académicas de alta calidad"""
    
    def __init__(self):
        self.logger = get_logger("AcademicPlots")
        self.viz_config = config.visualization
        self.output_dir = config.output_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar tema académico
        self.academic_theme = {
            'layout': {
                'font': {
                    'family': self.viz_config.font_family,
                    'size': self.viz_config.font_size
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': True,
                'legend': {
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.3,
                    'xanchor': 'center',
                    'x': 0.5
                }
            }
        }
    
    def create_time_series_analysis(self, df_pib: pd.DataFrame, df_iva: pd.DataFrame) -> go.Figure:
        """
        Crear análisis de series temporales académico
        
        Returns:
            Figure con análisis completo de series temporales
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'A) Evolución del PIB Español (2019-2023)',
                'B) Evolución del IVA Español (2019-2023)',
                'C) Descomposición Temporal PIB',
                'D) Descomposición Temporal IVA',
                'E) Análisis de Estacionalidad PIB',
                'F) Análisis de Estacionalidad IVA'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # A) Serie PIB
        fig.add_trace(
            go.Scatter(
                x=df_pib['fecha'],
                y=df_pib['valor'],
                mode='lines',
                name='PIB',
                line=dict(color=self.viz_config.color_primary, width=3),
                hovertemplate='<b>PIB</b><br>Fecha: %{x}<br>Valor: %{y:.1f} mil millones €<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Media móvil PIB
        if len(df_pib) >= 12:
            ma_pib = df_pib['valor'].rolling(12).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_pib['fecha'],
                    y=ma_pib,
                    mode='lines',
                    name='PIB MA(12)',
                    line=dict(color=self.viz_config.color_secondary, width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # B) Serie IVA
        fig.add_trace(
            go.Scatter(
                x=df_iva['fecha'],
                y=df_iva['valor'],
                mode='lines',
                name='IVA',
                line=dict(color=self.viz_config.color_accent, width=3),
                hovertemplate='<b>IVA</b><br>Fecha: %{x}<br>Valor: %{y:.1f} mil millones €<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Media móvil IVA
        if len(df_iva) >= 12:
            ma_iva = df_iva['valor'].rolling(12).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_iva['fecha'],
                    y=ma_iva,
                    mode='lines',
                    name='IVA MA(12)',
                    line=dict(color=self.viz_config.color_success, width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # C) Descomposición PIB
        if len(df_pib) >= 12:
            trend_pib = df_pib['valor'].rolling(12, center=True).mean()
            seasonal_pib = df_pib['valor'] - trend_pib
            
            fig.add_trace(
                go.Scatter(
                    x=df_pib['fecha'],
                    y=trend_pib,
                    mode='lines',
                    name='Tendencia PIB',
                    line=dict(color=self.viz_config.color_primary, width=3),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_pib['fecha'],
                    y=seasonal_pib,
                    mode='lines',
                    name='Componente Estacional PIB',
                    line=dict(color=self.viz_config.color_secondary, width=2),
                    yaxis='y3',
                    showlegend=False
                ),
                row=2, col=1, secondary_y=True
            )
        
        # D) Descomposición IVA
        if len(df_iva) >= 12:
            trend_iva = df_iva['valor'].rolling(12, center=True).mean()
            seasonal_iva = df_iva['valor'] - trend_iva
            
            fig.add_trace(
                go.Scatter(
                    x=df_iva['fecha'],
                    y=trend_iva,
                    mode='lines',
                    name='Tendencia IVA',
                    line=dict(color=self.viz_config.color_accent, width=3),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_iva['fecha'],
                    y=seasonal_iva,
                    mode='lines',
                    name='Componente Estacional IVA',
                    line=dict(color=self.viz_config.color_success, width=2),
                    yaxis='y4',
                    showlegend=False
                ),
                row=2, col=2, secondary_y=True
            )
        
        # E) Estacionalidad PIB
        df_pib_monthly = df_pib.copy()
        df_pib_monthly['mes'] = df_pib_monthly['fecha'].dt.month
        monthly_pib = df_pib_monthly.groupby('mes')['valor'].mean()
        
        fig.add_trace(
            go.Bar(
                x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                y=monthly_pib.values,
                name='PIB Promedio Mensual',
                marker_color=self.viz_config.color_primary,
                showlegend=False
            ),
            row=3, col=1
        )
        
        # F) Estacionalidad IVA
        df_iva_monthly = df_iva.copy()
        df_iva_monthly['mes'] = df_iva_monthly['fecha'].dt.month
        monthly_iva = df_iva_monthly.groupby('mes')['valor'].mean()
        
        fig.add_trace(
            go.Bar(
                x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                y=monthly_iva.values,
                name='IVA Promedio Mensual',
                marker_color=self.viz_config.color_accent,
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Configurar layout académico
        fig.update_layout(
            title={
                'text': '<b>Análisis Temporal Completo: PIB e IVA España (2019-2023)</b>',
                'x': 0.5,
                'font': {'size': self.viz_config.title_font_size}
            },
            height=1000,
            width=1400,
            template='plotly_white',
            showlegend=True
        )
        
        # Etiquetas de ejes
        fig.update_xaxes(title_text="Fecha", row=1, col=1)
        fig.update_xaxes(title_text="Fecha", row=1, col=2)
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_xaxes(title_text="Fecha", row=2, col=2)
        fig.update_xaxes(title_text="Mes", row=3, col=1)
        fig.update_xaxes(title_text="Mes", row=3, col=2)
        
        fig.update_yaxes(title_text="PIB (mil millones €)", row=1, col=1)
        fig.update_yaxes(title_text="IVA (mil millones €)", row=1, col=2)
        fig.update_yaxes(title_text="Tendencia PIB", row=2, col=1)
        fig.update_yaxes(title_text="Tendencia IVA", row=2, col=2)
        
        # Guardar visualización
        output_path = self.output_dir / "analisis_temporal_completo.html"
        fig.write_html(str(output_path))
        self.logger.log_visualization_created("Análisis Temporal Completo", str(output_path))
        
        return fig
    
    def create_correlation_analysis(self, df_pib: pd.DataFrame, df_iva: pd.DataFrame) -> go.Figure:
        """Crear análisis de correlación académico avanzado"""
        # Combinar datos
        df_combined = pd.merge(df_pib, df_iva, on='fecha', suffixes=('_pib', '_iva'))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'A) Correlación Cruzada PIB-IVA',
                'B) Scatter Plot con Regresión',
                'C) Correlación Móvil (12 meses)',
                'D) Análisis de Causalidad de Granger'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # A) Correlación cruzada
        lags = range(-12, 13)
        correlations = []
        for lag in lags:
            if lag < 0:
                corr = df_combined['valor_pib'].corr(df_combined['valor_iva'].shift(-lag))
            else:
                corr = df_combined['valor_pib'].shift(lag).corr(df_combined['valor_iva'])
            correlations.append(corr)
        
        colors = ['red' if corr < 0 else self.viz_config.color_primary for corr in correlations]
        fig.add_trace(
            go.Bar(
                x=lags,
                y=correlations,
                marker_color=colors,
                name='Correlación Cruzada',
                hovertemplate='Lag: %{x} meses<br>Correlación: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # B) Scatter plot con regresión
        fig.add_trace(
            go.Scatter(
                x=df_combined['valor_pib'],
                y=df_combined['valor_iva'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.viz_config.color_accent,
                    opacity=0.7
                ),
                name='Observaciones',
                hovertemplate='PIB: %{x:.1f}<br>IVA: %{y:.1f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Línea de regresión
        z = np.polyfit(df_combined['valor_pib'], df_combined['valor_iva'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df_combined['valor_pib'],
                y=p(df_combined['valor_pib']),
                mode='lines',
                line=dict(color='red', width=3),
                name=f'Regresión (R² = {np.corrcoef(df_combined["valor_pib"], df_combined["valor_iva"])[0,1]**2:.3f})',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # C) Correlación móvil
        window = 12
        rolling_corr = df_combined['valor_pib'].rolling(window).corr(df_combined['valor_iva'].rolling(window))
        
        fig.add_trace(
            go.Scatter(
                x=df_combined['fecha'][window:],
                y=rolling_corr[window:],
                mode='lines',
                line=dict(color=self.viz_config.color_secondary, width=3),
                name='Correlación Móvil',
                hovertemplate='Fecha: %{x}<br>Correlación: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # D) Tabla de estadísticas
        correlation_stats = [
            ['Correlación Pearson', f'{df_combined["valor_pib"].corr(df_combined["valor_iva"]):.4f}'],
            ['Correlación Spearman', f'{df_combined["valor_pib"].corr(df_combined["valor_iva"], method="spearman"):.4f}'],
            ['R² Lineal', f'{np.corrcoef(df_combined["valor_pib"], df_combined["valor_iva"])[0,1]**2:.4f}'],
            ['Lag Óptimo', f'{lags[np.argmax(np.abs(correlations))]} meses'],
            ['Correlación Máxima', f'{max(correlations, key=abs):.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Estadística</b>', '<b>Valor</b>'],
                    fill_color=self.viz_config.color_primary,
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=list(zip(*correlation_stats)),
                    fill_color='lavender',
                    font=dict(size=11)
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '<b>Análisis de Correlación PIB-IVA: Metodología Avanzada</b>',
                'x': 0.5,
                'font': {'size': self.viz_config.title_font_size}
            },
            height=800,
            width=1400,
            template='plotly_white'
        )
        
        output_path = self.output_dir / "analisis_correlacion.html"
        fig.write_html(str(output_path))
        self.logger.log_visualization_created("Análisis de Correlación", str(output_path))
        
        return fig
    
    def create_model_comparison_dashboard(self, model_results: Dict) -> go.Figure:
        """Crear dashboard académico de comparación de modelos"""
        # ...existing code for model comparison...
        # Implementar visualización completa de comparación de modelos
        pass
