"""
Dashboard Streamlit para Análisis PIB-IVA Colombia 2000-2024
Predicción IVA 2025 con modelos de Machine Learning y Series Temporales
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Imports para Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Imports para Series Temporales
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    from scipy.stats import boxcox
    import pmdarima as pm
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False

# Configuración de página
st.set_page_config(
    page_title="Análisis PIB-IVA Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar warnings
import warnings
warnings.filterwarnings('ignore')

class ColombiaEconomicAnalyzer:
    """Analizador económico específico para Colombia con modelos avanzados"""
    
    def __init__(self):
        self.models = {}
        self.ts_models = {}
        self.predictions = {}
        
    def generate_colombia_data(self):
        """Generar datos económicos realistas de Colombia 2000-2024"""
        # Período de análisis
        fechas = pd.date_range('2000-01-01', '2024-12-31', freq='A')
        n_years = len(fechas)
        
        # PIB Colombia (billones de pesos colombianos)
        pib_base = 250  # PIB base año 2000
        growth_rates = np.array([
            0.029, 0.015, 0.044, 0.039, 0.047, 0.067, 0.067, 0.035,
            0.017, 0.040, 0.040, 0.065, 0.054, 0.044, 0.032, 0.020,
            -0.068, 0.106, 0.035, 0.077, 0.050, 0.028, 0.075, 0.013, 0.033
        ])
        
        pib_values = [pib_base]
        for i in range(1, n_years):
            new_value = pib_values[-1] * (1 + growth_rates[i-1])
            pib_values.append(new_value)
        
        # Base del IVA Colombia
        base_iva_ratio = np.array([
            0.42, 0.43, 0.44, 0.45, 0.46, 0.48, 0.49, 0.47,
            0.45, 0.46, 0.48, 0.50, 0.52, 0.54, 0.53, 0.52,
            0.48, 0.51, 0.53, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60
        ])
        
        base_iva_values = np.array(pib_values) * base_iva_ratio
        
        # Tasa efectiva de IVA
        tasa_efectiva = np.array([
            0.155, 0.160, 0.163, 0.167, 0.169, 0.167, 0.168, 0.169,
            0.173, 0.163, 0.160, 0.158, 0.156, 0.157, 0.164, 0.163,
            0.146, 0.147, 0.147, 0.149, 0.150, 0.152, 0.155, 0.158, 0.160
        ])
        
        # IVA recaudado
        iva_values_calculated = base_iva_values * tasa_efectiva
        
        # Efectos económicos
        economic_shocks = np.array([
            1.0, 0.98, 1.02, 1.01, 1.03, 1.05, 1.04, 0.95,
            0.92, 1.08, 1.02, 1.06, 1.03, 1.01, 0.98, 0.96,
            0.85, 1.15, 1.05, 1.08, 1.02, 0.98, 1.12, 0.95, 1.03
        ])
        
        iva_values = iva_values_calculated * economic_shocks
        
        # Ruido realista
        pib_noise = np.random.normal(1, 0.02, n_years)
        iva_noise = np.random.normal(1, 0.03, n_years)
        base_noise = np.random.normal(1, 0.015, n_years)
        
        pib_values = np.array(pib_values) * pib_noise
        iva_values = iva_values * iva_noise
        base_iva_values = base_iva_values * base_noise
        
        # DataFrames
        df_pib = pd.DataFrame({
            'año': fechas.year,
            'fecha': fechas,
            'valor': pib_values,
            'crecimiento': [0] + list(np.diff(pib_values) / pib_values[:-1] * 100)
        })
        
        df_iva = pd.DataFrame({
            'año': fechas.year,
            'fecha': fechas,
            'valor': iva_values,
            'base_gravable': base_iva_values,
            'base_ratio_pib': base_iva_ratio * 100,
            'tasa_efectiva': tasa_efectiva * 100,
            'tasa_nominal': [16] * 17 + [19] * 8,
            'ratio_pib': iva_values / pib_values * 100,
            'eficiencia_recaudo': (tasa_efectiva / np.array([16] * 17 + [19] * 8) * 100) * 100
        })
        
        return df_pib, df_iva
    
    def prepare_features(self, df_pib, df_iva):
        """Preparar features para modelos predictivos"""
        df_combined = pd.merge(df_pib, df_iva, on='año', suffixes=('_pib', '_iva'))
        
        # Features económicas
        df_combined['pib_lag1'] = df_combined['valor_pib'].shift(1)
        df_combined['pib_lag2'] = df_combined['valor_pib'].shift(2)
        df_combined['iva_lag1'] = df_combined['valor_iva'].shift(1)
        df_combined['pib_growth'] = df_combined['crecimiento']
        df_combined['pib_ma3'] = df_combined['valor_pib'].rolling(3).mean()
        df_combined['ratio_trend'] = df_combined['ratio_pib'].rolling(3).mean()
        
        # Features temporales
        df_combined['año_norm'] = (df_combined['año'] - 2000) / 24
        df_combined['decada'] = (df_combined['año'] - 2000) // 10
        
        return df_combined
    
    def train_models(self, df_combined):
        """Entrenar múltiples modelos predictivos"""
        if not ML_AVAILABLE:
            st.error("❌ Scikit-learn no disponible para entrenamiento de modelos")
            return {}
            
        # Preparar datos
        features = ['valor_pib', 'pib_lag1', 'pib_lag2', 'iva_lag1', 'pib_growth', 
                   'pib_ma3', 'año_norm', 'decada']
        
        df_model = df_combined[features + ['valor_iva']].dropna()
        X = df_model[features]
        y = df_model['valor_iva']
        
        # División temporal (últimos 5 años para test)
        split_idx = len(X) - 5
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Modelos
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Regresión Lineal': LinearRegression()
        }
        
        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'modelo': model,
                    'rmse': rmse,
                    'r2': r2,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            except Exception as e:
                st.error(f"Error entrenando {name}: {e}")
        
        self.models = results
        return results
    
    def predict_2025(self, df_pib, df_iva, pib_2025_scenario):
        """Predecir IVA 2025 basado en escenario PIB"""
        if not self.models:
            # Predicción simplificada sin ML
            ratio_promedio = df_iva['ratio_pib'].mean() / 100
            prediccion_simple = pib_2025_scenario * ratio_promedio
            return {'Predicción Simple': prediccion_simple}
            
        df_combined = self.prepare_features(df_pib, df_iva)
        
        # Crear features para 2025
        last_row = df_combined.iloc[-1].copy()
        
        # Actualizar para 2025
        features_2025 = {
            'valor_pib': pib_2025_scenario,
            'pib_lag1': last_row['valor_pib'],
            'pib_lag2': df_combined.iloc[-2]['valor_pib'],
            'iva_lag1': last_row['valor_iva'],
            'pib_growth': (pib_2025_scenario - last_row['valor_pib']) / last_row['valor_pib'] * 100,
            'pib_ma3': np.mean([pib_2025_scenario, last_row['valor_pib'], df_combined.iloc[-2]['valor_pib']]),
            'año_norm': (2025 - 2000) / 24,
            'decada': 2
        }
        
        X_2025 = pd.DataFrame([features_2025])
        
        predictions_2025 = {}
        for name, result in self.models.items():
            try:
                pred = result['modelo'].predict(X_2025)[0]
                predictions_2025[name] = pred
            except Exception as e:
                st.error(f"Error prediciendo con {name}: {e}")
        
        return predictions_2025

    def apply_boxcox_transformation(self, series, name="Serie"):
        """Aplicar transformación Box-Cox para estabilizar varianza"""
        if not TS_AVAILABLE:
            return series, 1.0, "Box-Cox no disponible"
        
        try:
            # Asegurar valores positivos
            series_positive = series + abs(series.min()) + 1 if series.min() <= 0 else series
            
            # Encontrar lambda óptimo
            transformed_data, fitted_lambda = boxcox(series_positive)
            
            # Interpretación del lambda
            if abs(fitted_lambda) < 0.1:
                interpretation = "Transformación logarítmica"
            elif abs(fitted_lambda - 0.5) < 0.1:
                interpretation = "Transformación raíz cuadrada"
            elif abs(fitted_lambda - 1) < 0.1:
                interpretation = "Sin transformación necesaria"
            else:
                interpretation = f"Transformación Box-Cox (λ={fitted_lambda:.3f})"
            
            return transformed_data, fitted_lambda, interpretation
            
        except Exception as e:
            st.warning(f"Error en Box-Cox para {name}: {e}")
            return series, 1.0, "Error en transformación"
    
    def fit_arima_models(self, df_pib, df_iva):
        """Ajustar modelos ARIMA univariados"""
        if not TS_AVAILABLE:
            return {}
        
        results = {}
        
        try:
            # Preparar series temporales
            ts_iva = df_iva.set_index('fecha')['valor']
            ts_pib = df_pib.set_index('fecha')['valor']
            
            # Box-Cox para IVA
            iva_transformed, lambda_iva, interp_iva = self.apply_boxcox_transformation(ts_iva, "IVA")
            ts_iva_bc = pd.Series(iva_transformed, index=ts_iva.index)
            
            # Auto-ARIMA para IVA
            auto_arima = pm.auto_arima(
                ts_iva_bc,
                seasonal=True,
                m=1,  # frecuencia anual
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3, max_q=3, max_P=2, max_Q=2
            )
            
            results['ARIMA_IVA'] = {
                'model': auto_arima,
                'series': ts_iva_bc,
                'original_series': ts_iva,
                'lambda': lambda_iva,
                'transformation': interp_iva,
                'order': auto_arima.order,
                'seasonal_order': auto_arima.seasonal_order,
                'aic': auto_arima.aic(),
                'bic': auto_arima.bic()
            }
            
            # ARIMA para PIB (variable exógena)
            pib_transformed, lambda_pib, interp_pib = self.apply_boxcox_transformation(ts_pib, "PIB")
            ts_pib_bc = pd.Series(pib_transformed, index=ts_pib.index)
            
            auto_arima_pib = pm.auto_arima(
                ts_pib_bc,
                seasonal=True,
                m=1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            results['ARIMA_PIB'] = {
                'model': auto_arima_pib,
                'series': ts_pib_bc,
                'original_series': ts_pib,
                'lambda': lambda_pib,
                'transformation': interp_pib,
                'order': auto_arima_pib.order,
                'aic': auto_arima_pib.aic()
            }
            
        except Exception as e:
            st.error(f"Error en modelos ARIMA: {e}")
        
        return results
    
    def fit_sarimax_model(self, df_pib, df_iva):
        """Ajustar modelo SARIMAX con PIB como variable exógena"""
        if not TS_AVAILABLE:
            return {}
        
        try:
            # Preparar datos
            ts_iva = df_iva.set_index('fecha')['valor']
            ts_pib = df_pib.set_index('fecha')['valor']
            
            # Transformaciones Box-Cox
            iva_transformed, lambda_iva, _ = self.apply_boxcox_transformation(ts_iva)
            pib_transformed, lambda_pib, _ = self.apply_boxcox_transformation(ts_pib)
            
            # SARIMAX con PIB como exógena
            model_sarimax = SARIMAX(
                iva_transformed,
                exog=pib_transformed,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_sarimax = model_sarimax.fit(disp=False)
            
            return {
                'model': fitted_sarimax,
                'iva_series': iva_transformed,
                'pib_series': pib_transformed,
                'lambda_iva': lambda_iva,
                'lambda_pib': lambda_pib,
                'aic': fitted_sarimax.aic,
                'bic': fitted_sarimax.bic,
                'llf': fitted_sarimax.llf
            }
            
        except Exception as e:
            st.error(f"Error en modelo SARIMAX: {e}")
            return {}
    
    def fit_var_model(self, df_pib, df_iva):
        """Ajustar modelo VAR multivariado"""
        if not TS_AVAILABLE:
            return {}
        
        try:
            # Preparar datos multivariados
            ts_data = pd.DataFrame({
                'IVA': df_iva.set_index('fecha')['valor'],
                'PIB': df_pib.set_index('fecha')['valor']
            })
            
            # Transformaciones Box-Cox
            for col in ts_data.columns:
                transformed, _, _ = self.apply_boxcox_transformation(ts_data[col])
                ts_data[col] = transformed
            
            # Diferenciar para estacionariedad
            ts_data_diff = ts_data.diff().dropna()
            
            # Selección automática de lags
            var_model = VAR(ts_data_diff)
            lag_order = var_model.select_order(maxlags=4)
            optimal_lags = lag_order.aic
            
            # Ajustar modelo VAR
            var_fitted = var_model.fit(optimal_lags)
            
            return {
                'model': var_fitted,
                'data': ts_data_diff,
                'original_data': ts_data,
                'optimal_lags': optimal_lags,
                'aic': var_fitted.aic,
                'bic': var_fitted.bic
            }
            
        except Exception as e:
            st.error(f"Error en modelo VAR: {e}")
            return {}
    
    def forecast_ts_models(self, ts_results, periods=1):
        """Realizar pronósticos con modelos de series temporales"""
        if not TS_AVAILABLE or not ts_results:
            return {}
        
        forecasts = {}
        
        try:
            # Pronóstico ARIMA
            if 'ARIMA_IVA' in ts_results:
                arima_model = ts_results['ARIMA_IVA']['model']
                arima_forecast = arima_model.predict(n_periods=periods)
                
                # Transformación inversa Box-Cox
                lambda_val = ts_results['ARIMA_IVA']['lambda']
                if lambda_val != 1.0:
                    if lambda_val == 0:
                        arima_forecast_orig = np.exp(arima_forecast)
                    else:
                        arima_forecast_orig = (arima_forecast * lambda_val + 1) ** (1/lambda_val)
                else:
                    arima_forecast_orig = arima_forecast
                
                forecasts['ARIMA'] = arima_forecast_orig[0] if len(arima_forecast_orig) > 0 else None
            
            # Pronóstico SARIMAX
            if 'SARIMAX' in ts_results:
                # Para SARIMAX necesitamos proyectar PIB 2025
                # Usar crecimiento promedio histórico
                pib_growth_avg = 0.035  # 3.5% promedio
                last_pib = ts_results['SARIMAX']['pib_series'][-1]
                pib_2025_proj = last_pib * (1 + pib_growth_avg)
                
                sarimax_forecast = ts_results['SARIMAX']['model'].forecast(
                    steps=periods, 
                    exog=np.array([[pib_2025_proj]])
                )
                
                # Transformación inversa
                lambda_iva = ts_results['SARIMAX']['lambda_iva']
                if lambda_iva != 1.0:
                    if lambda_iva == 0:
                        sarimax_forecast_orig = np.exp(sarimax_forecast)
                    else:
                        sarimax_forecast_orig = (sarimax_forecast * lambda_iva + 1) ** (1/lambda_iva)
                else:
                    sarimax_forecast_orig = sarimax_forecast
                
                forecasts['SARIMAX'] = sarimax_forecast_orig[0] if len(sarimax_forecast_orig) > 0 else None
            
            # Pronóstico VAR
            if 'VAR' in ts_results:
                var_forecast = ts_results['VAR']['model'].forecast(
                    ts_results['VAR']['data'].values[-ts_results['VAR']['optimal_lags']:], 
                    steps=periods
                )
                
                # El pronóstico VAR está en diferencias, necesitamos acumular
                last_values = ts_results['VAR']['original_data'].iloc[-1].values
                var_forecast_level = last_values[0] + var_forecast[0, 0]  # IVA
                
                forecasts['VAR'] = var_forecast_level
                
        except Exception as e:
            st.error(f"Error en pronósticos: {e}")
        
        return forecasts

# Instanciar analizador
@st.cache_data
def load_data():
    analyzer = ColombiaEconomicAnalyzer()
    df_pib, df_iva = analyzer.generate_colombia_data()
    return analyzer, df_pib, df_iva

def main():
    """Aplicación principal de Streamlit"""
    
    st.title("🇨🇴 Análisis PIB-IVA Colombia 2000-2024")
    st.markdown("### Predicción IVA 2025 con Machine Learning y Series Temporales")
    
    # Cargar datos
    try:
        analyzer, df_pib, df_iva = load_data()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()
    
    # Presentación inicial con análisis de series
    st.markdown("---")
    st.header("📊 Análisis Inicial de Series Económicas")
    
    # Crear visualización inicial simplificada
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 PIB Colombia 2000-2024")
        
        fig_pib = go.Figure()
        fig_pib.add_trace(go.Scatter(
            x=df_pib['año'],
            y=df_pib['valor'],
            mode='lines+markers',
            name='PIB',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Año: %{x}<br>PIB: %{y:.1f} billones COP<extra></extra>'
        ))
        
        # Marcar eventos importantes
        eventos = {2008: "Crisis", 2020: "COVID-19"}
        for año, evento in eventos.items():
            fig_pib.add_vline(x=año, line_dash="dash", line_color="red", 
                             annotation_text=evento)
        
        fig_pib.update_layout(
            title="Evolución PIB Colombia",
            xaxis_title="Año",
            yaxis_title="PIB (billones COP)",
            height=400
        )
        st.plotly_chart(fig_pib, use_container_width=True)
        
        # Estadísticas PIB
        st.info(f"""
        **PIB Colombia:**
        - 2000: {df_pib['valor'].iloc[0]:.1f} billones COP
        - 2024: {df_pib['valor'].iloc[-1]:.1f} billones COP
        - Crecimiento promedio: {df_pib['crecimiento'][1:].mean():.1f}%
        - Mayor crecimiento: {df_pib['crecimiento'][1:].max():.1f}%
        - Mayor contracción: {df_pib['crecimiento'][1:].min():.1f}%
        """)
    
    with col2:
        st.subheader("💰 IVA Colombia 2000-2024")
        
        fig_iva = go.Figure()
        fig_iva.add_trace(go.Scatter(
            x=df_iva['año'],
            y=df_iva['valor'],
            mode='lines+markers',
            name='IVA',
            line=dict(color='#ff7f0e', width=3),
            hovertemplate='Año: %{x}<br>IVA: %{y:.1f} billones COP<extra></extra>'
        ))
        
        # Marcar reformas tributarias
        reformas = {2005: "Reforma Uribe", 2017: "IVA 19%"}
        for año, reforma in reformas.items():
            fig_iva.add_vline(x=año, line_dash="dash", line_color="orange", 
                             annotation_text=reforma)
        
        fig_iva.update_layout(
            title="Evolución IVA Colombia",
            xaxis_title="Año",
            yaxis_title="IVA (billones COP)",
            height=400
        )
        st.plotly_chart(fig_iva, use_container_width=True)
        
        # Estadísticas IVA
        st.info(f"""
        **IVA Colombia:**
        - 2000: {df_iva['valor'].iloc[0]:.1f} billones COP
        - 2024: {df_iva['valor'].iloc[-1]:.1f} billones COP
        - Ratio IVA/PIB: {df_iva['ratio_pib'].iloc[-1]:.1f}%
        - Base gravable: {df_iva['base_ratio_pib'].iloc[-1]:.1f}% del PIB
        - Eficiencia: {df_iva['eficiencia_recaudo'].iloc[-1]:.1f}%
        """)
    
    # Correlación inicial
    st.subheader("🔗 Correlación PIB-IVA")
    correlation = np.corrcoef(df_pib['valor'], df_iva['valor'])[0,1]
    
    fig_corr_inicial = go.Figure()
    fig_corr_inicial.add_trace(go.Scatter(
        x=df_pib['valor'],
        y=df_iva['valor'],
        mode='markers',
        marker=dict(
            size=10,
            color=df_iva['año'],
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Año")
        ),
        text=[f"Año: {year}" for year in df_iva['año']],
        hovertemplate='PIB: %{x:.1f}<br>IVA: %{y:.1f}<br>%{text}<extra></extra>'
    ))
    
    # Línea de tendencia
    z = np.polyfit(df_pib['valor'], df_iva['valor'], 1)
    p = np.poly1d(z)
    fig_corr_inicial.add_trace(go.Scatter(
        x=df_pib['valor'],
        y=p(df_pib['valor']),
        mode='lines',
        name='Tendencia',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig_corr_inicial.update_layout(
        title=f"Correlación PIB-IVA Colombia (r = {correlation:.3f})",
        xaxis_title="PIB (billones COP)",
        yaxis_title="IVA (billones COP)",
        height=400
    )
    st.plotly_chart(fig_corr_inicial, use_container_width=True)
    
    st.success(f"""
    **🎯 Insights Principales:**
    - Correlación PIB-IVA: **{correlation:.3f}** (muy fuerte)
    - Crecimiento IVA: {((df_iva['valor'].iloc[-1]/df_iva['valor'].iloc[0])**0.04167-1)*100:.1f}% promedio anual
    - Base gravable actual: {df_iva['base_ratio_pib'].iloc[-1]:.1f}% del PIB
    - Eficiencia recaudo: {df_iva['eficiencia_recaudo'].iloc[-1]:.1f}% del potencial
    """)
    
    # Sidebar configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Escenarios PIB 2025
    pib_actual = df_pib['valor'].iloc[-1]
    scenario = st.sidebar.selectbox(
        "Escenario PIB 2025:",
        ["Conservador (2%)", "Moderado (3.5%)", "Optimista (5%)", "Personalizado"]
    )
    
    if scenario == "Conservador (2%)":
        pib_2025 = pib_actual * 1.02
    elif scenario == "Moderado (3.5%)":
        pib_2025 = pib_actual * 1.035
    elif scenario == "Optimista (5%)":
        pib_2025 = pib_actual * 1.05
    else:
        crecimiento = st.sidebar.slider("Crecimiento PIB 2025 (%)", -2.0, 8.0, 3.5, 0.1)
        pib_2025 = pib_actual * (1 + crecimiento/100)
    
    st.sidebar.metric("PIB 2025 Proyectado", f"{pib_2025:.1f} billones COP")
    
    # Tabs principales expandidos
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Análisis Detallado", 
        "📈 Box-Cox & Transformaciones", 
        "🔄 ARIMA & SARIMA", 
        "🌐 SARIMAX & VAR",
        "🤖 Machine Learning", 
        "🔮 Predicción 2025"
    ])
    
    with tab1:
        st.header("📊 Análisis Histórico Detallado")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("PIB 2024", f"{df_pib['valor'].iloc[-1]:.1f} billones")
        with col2:
            st.metric("IVA 2024", f"{df_iva['valor'].iloc[-1]:.1f} billones")
        with col3:
            st.metric("Base Gravable", f"{df_iva['base_gravable'].iloc[-1]:.1f} billones")
        with col4:
            st.metric("Eficiencia", f"{df_iva['eficiencia_recaudo'].iloc[-1]:.1f}%")
        
        # Tabla de datos recientes
        st.subheader("📋 Datos Históricos Recientes")
        
        recent_data = pd.merge(
            df_pib[['año', 'valor', 'crecimiento']].tail(10),
            df_iva[['año', 'valor', 'base_gravable', 'eficiencia_recaudo']].tail(10),
            on='año'
        )
        recent_data.columns = ['Año', 'PIB (billones)', 'Crecimiento PIB (%)', 
                              'IVA (billones)', 'Base Gravable (billones)', 'Eficiencia (%)']
        
        # Formatear números
        for col in ['PIB (billones)', 'IVA (billones)', 'Base Gravable (billones)']:
            recent_data[col] = recent_data[col].round(1)
        for col in ['Crecimiento PIB (%)', 'Eficiencia (%)']:
            recent_data[col] = recent_data[col].round(1)
        
        st.dataframe(recent_data, use_container_width=True)
    
    with tab2:
        st.header("📈 Análisis Box-Cox y Transformaciones")
        
        if TS_AVAILABLE:
            st.subheader("🔧 Transformaciones para Estabilizar Varianza")
            
            # Aplicar Box-Cox a ambas series
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 IVA - Transformación Box-Cox")
                
                # Box-Cox para IVA
                iva_transformed, lambda_iva, interp_iva = analyzer.apply_boxcox_transformation(df_iva['valor'], "IVA")
                
                # Comparar original vs transformada
                fig_bc_iva = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('IVA Original', 'IVA Transformada (Box-Cox)')
                )
                
                fig_bc_iva.add_trace(
                    go.Scatter(x=df_iva['año'], y=df_iva['valor'], mode='lines+markers', name='Original'),
                    row=1, col=1
                )
                
                fig_bc_iva.add_trace(
                    go.Scatter(x=df_iva['año'], y=iva_transformed, mode='lines+markers', name='Transformada'),
                    row=2, col=1
                )
                
                fig_bc_iva.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_bc_iva, use_container_width=True)
                
                st.info(f"""
                **Transformación IVA:**
                - λ (lambda): {lambda_iva:.3f}
                - Interpretación: {interp_iva}
                - Efecto: Estabiliza varianza temporal
                """)
            
            with col2:
                st.subheader("📈 PIB - Transformación Box-Cox")
                
                # Box-Cox para PIB
                pib_transformed, lambda_pib, interp_pib = analyzer.apply_boxcox_transformation(df_pib['valor'], "PIB")
                
                # Comparar original vs transformada
                fig_bc_pib = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('PIB Original', 'PIB Transformado (Box-Cox)')
                )
                
                fig_bc_pib.add_trace(
                    go.Scatter(x=df_pib['año'], y=df_pib['valor'], mode='lines+markers', name='Original'),
                    row=1, col=1
                )
                
                fig_bc_pib.add_trace(
                    go.Scatter(x=df_pib['año'], y=pib_transformed, mode='lines+markers', name='Transformada'),
                    row=2, col=1
                )
                
                fig_bc_pib.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_bc_pib, use_container_width=True)
                
                st.info(f"""
                **Transformación PIB:**
                - λ (lambda): {lambda_pib:.3f}
                - Interpretación: {interp_pib}
                - Efecto: Linealiza tendencia
                """)
            
            # Test de estacionariedad
            st.subheader("📊 Análisis de Estacionariedad")
            
            try:
                from statsmodels.tsa.stattools import adfuller
                
                # Test ADF para IVA
                adf_iva = adfuller(iva_transformed)
                # Test ADF para PIB
                adf_pib = adfuller(pib_transformed)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Test ADF - IVA",
                        f"{adf_iva[1]:.4f}",
                        "Estacionaria" if adf_iva[1] < 0.05 else "No estacionaria"
                    )
                
                with col2:
                    st.metric(
                        "Test ADF - PIB", 
                        f"{adf_pib[1]:.4f}",
                        "Estacionaria" if adf_pib[1] < 0.05 else "No estacionaria"
                    )
                
            except Exception as e:
                st.warning(f"Error en test de estacionariedad: {e}")
                
        else:
            st.error("⚠️ Statsmodels no disponible. Instala con: pip install statsmodels")
    
    with tab3:
        st.header("🔄 Modelos ARIMA y SARIMA")
        
        if TS_AVAILABLE:
            # Ajustar modelos ARIMA
            with st.spinner("Ajustando modelos ARIMA..."):
                arima_results = analyzer.fit_arima_models(df_pib, df_iva)
            
            if arima_results:
                st.subheader("📊 Resultados ARIMA")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'ARIMA_IVA' in arima_results:
                        st.subheader("💰 ARIMA - IVA")
                        
                        arima_iva = arima_results['ARIMA_IVA']
                        
                        st.info(f"""
                        **Modelo ARIMA IVA:**
                        - Orden: {arima_iva['order']}
                        - Orden estacional: {arima_iva['seasonal_order']}
                        - AIC: {arima_iva['aic']:.2f}
                        - BIC: {arima_iva['bic']:.2f}
                        - Transformación: {arima_iva['transformation']}
                        """)
                        
                        # Residuos
                        residuos = arima_iva['model'].resid()
                        
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            x=list(range(len(residuos))),
                            y=residuos,
                            mode='lines',
                            name='Residuos'
                        ))
                        fig_resid.update_layout(title="Residuos ARIMA IVA")
                        st.plotly_chart(fig_resid, use_container_width=True)
                
                with col2:
                    if 'ARIMA_PIB' in arima_results:
                        st.subheader("📈 ARIMA - PIB")
                        
                        arima_pib = arima_results['ARIMA_PIB']
                        
                        st.info(f"""
                        **Modelo ARIMA PIB:**
                        - Orden: {arima_pib['order']}
                        - AIC: {arima_pib['aic']:.2f}
                        - Transformación: {arima_pib['transformation']}
                        """)
                        
                        # Fitted vs Actual
                        fitted = arima_pib['model'].fittedvalues()
                        
                        fig_fit = go.Figure()
                        fig_fit.add_trace(go.Scatter(
                            x=list(range(len(arima_pib['series']))),
                            y=arima_pib['series'],
                            mode='lines',
                            name='Actual'
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=list(range(len(fitted))),
                            y=fitted,
                            mode='lines',
                            name='Ajustado'
                        ))
                        fig_fit.update_layout(title="ARIMA PIB: Actual vs Ajustado")
                        st.plotly_chart(fig_fit, use_container_width=True)
                        
                # Diagnósticos
                st.subheader("🔍 Diagnósticos del Modelo")
                
                if 'ARIMA_IVA' in arima_results:
                    residuos = arima_results['ARIMA_IVA']['model'].resid()
                    
                    # Test Ljung-Box
                    try:
                        ljung_box = acorr_ljungbox(residuos, lags=10, return_df=True)
                        
                        st.write("**Test Ljung-Box (Autocorrelación de Residuos):**")
                        st.dataframe(ljung_box.round(4))
                        
                        if ljung_box['lb_pvalue'].iloc[-1] > 0.05:
                            st.success("✅ No hay autocorrelación en residuos (p > 0.05)")
                        else:
                            st.warning("⚠️ Posible autocorrelación en residuos (p < 0.05)")
                            
                    except Exception as e:
                        st.warning(f"Error en diagnósticos: {e}")
            else:
                st.error("No se pudieron ajustar modelos ARIMA")
        else:
            st.error("⚠️ Instala statsmodels y pmdarima para modelos ARIMA")
    
    with tab4:
        st.header("🌐 Modelos SARIMAX y VAR")
        
        if TS_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 SARIMAX (PIB como Exógena)")
                
                with st.spinner("Ajustando SARIMAX..."):
                    sarimax_results = analyzer.fit_sarimax_model(df_pib, df_iva)
                
                if sarimax_results:
                    st.info(f"""
                    **Modelo SARIMAX:**
                    - AIC: {sarimax_results['aic']:.2f}
                    - BIC: {sarimax_results['bic']:.2f}
                    - Log-Likelihood: {sarimax_results['llf']:.2f}
                    - Variable exógena: PIB transformado
                    """)
                    
                    # Residuos SARIMAX
                    residuos_sarimax = sarimax_results['model'].resid
                    
                    fig_sarimax_resid = go.Figure()
                    fig_sarimax_resid.add_trace(go.Scatter(
                        x=list(range(len(residuos_sarimax))),
                        y=residuos_sarimax,
                        mode='lines',
                        name='Residuos SARIMAX'
                    ))
                    fig_sarimax_resid.update_layout(title="Residuos SARIMAX")
                    st.plotly_chart(fig_sarimax_resid, use_container_width=True)
                else:
                    st.error("Error ajustando SARIMAX")
            
            with col2:
                st.subheader("🔄 VAR (Vector Autoregresivo)")
                
                with st.spinner("Ajustando VAR..."):
                    var_results = analyzer.fit_var_model(df_pib, df_iva)
                
                if var_results:
                    st.info(f"""
                    **Modelo VAR:**
                    - Lags óptimos: {var_results['optimal_lags']}
                    - AIC: {var_results['aic']:.2f}
                    - BIC: {var_results['bic']:.2f}
                    - Variables: IVA, PIB (multivariado)
                    """)
                    
                    # Impulse Response Functions
                    try:
                        irf = var_results['model'].irf(periods=10)
                        
                        fig_irf = go.Figure()
                        
                        # IRF de PIB a IVA
                        irf_pib_to_iva = irf.irfs[:, 0, 1]  # IVA response to PIB shock
                        
                        fig_irf.add_trace(go.Scatter(
                            x=list(range(len(irf_pib_to_iva))),
                            y=irf_pib_to_iva,
                            mode='lines+markers',
                            name='Respuesta IVA a shock PIB'
                        ))
                        
                        fig_irf.update_layout(
                            title="Función de Respuesta al Impulso",
                            xaxis_title="Períodos",
                            yaxis_title="Respuesta"
                        )
                        st.plotly_chart(fig_irf, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Error en IRF: {e}")
                else:
                    st.error("Error ajustando VAR")
        else:
            st.error("⚠️ Instala statsmodels para SARIMAX y VAR")
    
    with tab5:
        st.header("🤖 Modelos de Machine Learning")
        
        if ML_AVAILABLE:
            # Entrenar modelos
            df_combined = analyzer.prepare_features(df_pib, df_iva)
            results = analyzer.train_models(df_combined)
            
            if results:
                # Mostrar resultados
                st.subheader("📊 Performance de Modelos")
                
                metrics_df = pd.DataFrame({
                    'Modelo': list(results.keys()),
                    'RMSE': [results[k]['rmse'] for k in results.keys()],
                    'R²': [results[k]['r2'] for k in results.keys()]
                })
                
                st.dataframe(metrics_df.round(4), use_container_width=True)
                
                # Visualización comparativa
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Bar(
                        x=metrics_df['Modelo'],
                        y=metrics_df['R²'],
                        marker_color='lightblue'
                    ))
                    fig_r2.update_layout(title="R² por Modelo", yaxis_title="R²")
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(go.Bar(
                        x=metrics_df['Modelo'],
                        y=metrics_df['RMSE'],
                        marker_color='lightcoral'
                    ))
                    fig_rmse.update_layout(title="RMSE por Modelo", yaxis_title="RMSE")
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Predicciones vs valores reales
                st.subheader("🎯 Predicciones vs Valores Reales")
                
                best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(results[best_model]['y_test']))),
                    y=results[best_model]['y_test'],
                    mode='lines+markers',
                    name='Valores Reales',
                    line=dict(color='blue')
                ))
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(results[best_model]['y_pred']))),
                    y=results[best_model]['y_pred'],
                    mode='lines+markers',
                    name='Predicciones',
                    line=dict(color='red', dash='dash')
                ))
                fig_pred.update_layout(title=f"Mejor Modelo: {best_model}")
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.error("❌ No se pudieron entrenar los modelos")
        else:
            st.warning("⚠️ Instala scikit-learn para funcionalidad completa de ML")
            st.code("pip install scikit-learn")
    
    with tab6:
        st.header("🔮 Predicción IVA 2025 - Todos los Modelos")
        
        # Compilar todos los modelos
        all_predictions = {}
        
        # Predicciones ML
        if ML_AVAILABLE:
            ml_predictions = analyzer.predict_2025(df_pib, df_iva, pib_2025)
            all_predictions.update(ml_predictions)
        
        # Predicciones Series Temporales
        if TS_AVAILABLE:
            # Recopilar modelos TS
            ts_models = {}
            
            # ARIMA
            arima_results = analyzer.fit_arima_models(df_pib, df_iva)
            if arima_results:
                ts_models.update(arima_results)
            
            # SARIMAX
            sarimax_results = analyzer.fit_sarimax_model(df_pib, df_iva)
            if sarimax_results:
                ts_models['SARIMAX'] = sarimax_results
            
            # VAR
            var_results = analyzer.fit_var_model(df_pib, df_iva)
            if var_results:
                ts_models['VAR'] = var_results
            
            # Pronósticos TS
            ts_forecasts = analyzer.forecast_ts_models(ts_models, periods=1)
            all_predictions.update(ts_forecasts)
        
        if all_predictions:
            st.subheader("📊 Comparación de Todos los Modelos")
            
            # Tabla comparativa
            pred_comparison = []
            for model, pred in all_predictions.items():
                if pred is not None:
                    growth = ((pred / df_iva['valor'].iloc[-1]) - 1) * 100
                    pred_comparison.append({
                        'Modelo': model,
                        'Tipo': 'ML' if model in ['Random Forest', 'Gradient Boosting', 'Regresión Lineal'] else 'Series Temporales',
                        'IVA 2025 (billones)': f"{pred:.1f}",
                        'Crecimiento vs 2024': f"{growth:+.1f}%",
                        'Ratio IVA/PIB': f"{(pred/pib_2025)*100:.1f}%"
                    })
            
            comparison_df = pd.DataFrame(pred_comparison)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualización comparativa
            fig_comparison = go.Figure()
            
            # Histórico
            fig_comparison.add_trace(go.Scatter(
                x=df_iva['año'],
                y=df_iva['valor'],
                mode='lines+markers',
                name='IVA Histórico',
                line=dict(color='blue', width=3)
            ))
            
            # Predicciones por tipo
            ml_preds = [v for k, v in all_predictions.items() if k in ['Random Forest', 'Gradient Boosting', 'Regresión Lineal'] and v is not None]
            ts_preds = [v for k, v in all_predictions.items() if k not in ['Random Forest', 'Gradient Boosting', 'Regresión Lineal'] and v is not None]
            
            if ml_preds:
                avg_ml = np.mean(ml_preds)
                fig_comparison.add_trace(go.Scatter(
                    x=[2024, 2025],
                    y=[df_iva['valor'].iloc[-1], avg_ml],
                    mode='lines+markers',
                    name='Promedio ML',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=12)
                ))
            
            if ts_preds:
                avg_ts = np.mean(ts_preds)
                fig_comparison.add_trace(go.Scatter(
                    x=[2024, 2025],
                    y=[df_iva['valor'].iloc[-1], avg_ts],
                    mode='lines+markers',
                    name='Promedio Series Temporales',
                    line=dict(color='green', width=3, dash='dot'),
                    marker=dict(size=12)
                ))
            
            fig_comparison.update_layout(
                title="Predicciones IVA 2025: ML vs Series Temporales",
                xaxis_title="Año",
                yaxis_title="IVA (billones COP)",
                height=500
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Estadísticas de consenso
            all_valid_preds = [v for v in all_predictions.values() if v is not None]
            if all_valid_preds:
                consensus = np.mean(all_valid_preds)
                std_dev = np.std(all_valid_preds)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Consenso (Promedio)", f"{consensus:.1f} billones")
                col2.metric("Desviación Estándar", f"{std_dev:.1f} billones")
                col3.metric("Rango de Predicciones", f"{max(all_valid_preds) - min(all_valid_preds):.1f} billones")
                
                st.success(f"""
                **🎯 Consenso de Modelos 2025:**
                - **Predicción consenso**: {consensus:.1f} billones COP
                - **Crecimiento esperado**: {((consensus/df_iva['valor'].iloc[-1])-1)*100:+.1f}%
                - **Confianza**: ±{std_dev:.1f} billones (1σ)
                - **Modelos utilizados**: {len(all_valid_preds)} diferentes enfoques
                """)
        else:
            st.error("No se pudieron generar predicciones")

if __name__ == "__main__":
    main()
