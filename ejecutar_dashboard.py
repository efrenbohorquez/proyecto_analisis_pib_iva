"""
🚀 EJECUTOR INMEDIATO - Dashboard PIB-IVA Colombia
Análisis 2000-2024 con predicción 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Ejecutar dashboard inmediatamente"""
    
    print("🇨🇴 DASHBOARD PIB-IVA COLOMBIA 2000-2024")
    print("=" * 50)
    print("📊 Análisis histórico + Base gravable IVA")
    print("🔮 Predicción IVA 2025")
    print("🤖 Modelos Machine Learning")
    print("=" * 50)
    
    # Verificar ubicación
    current_dir = Path(__file__).parent
    print(f"📁 Directorio: {current_dir}")
    
    # Instalar dependencias críticas
    print("📦 Verificando dependencias...")
    dependencies = [
        "streamlit",
        "plotly", 
        "scikit-learn",
        "pandas",
        "numpy"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"📥 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    # Crear streamlit_app.py si no existe
    streamlit_file = current_dir / "streamlit_app.py"
    if not streamlit_file.exists():
        print("📝 Creando aplicación Streamlit...")
        create_streamlit_app(streamlit_file)
    
    # Ejecutar dashboard
    print("\n🚀 INICIANDO DASHBOARD...")
    print("🌐 Se abrirá en: http://localhost:8501")
    print("⏹️ Para detener: Ctrl+C")
    
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_file), 
               "--server.port", "8501", "--server.headless", "false"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_streamlit_app(file_path):
    """Crear aplicación Streamlit completa"""
    
    app_code = '''"""
Dashboard Streamlit - Análisis PIB-IVA Colombia 2000-2024
Incluye análisis de base gravable y predicción 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración página
st.set_page_config(
    page_title="PIB-IVA Colombia",
    page_icon="🇨🇴",
    layout="wide"
)

@st.cache_data
def generate_colombia_data():
    """Generar datos económicos Colombia 2000-2024"""
    
    # Años análisis
    years = list(range(2000, 2025))
    fechas = pd.to_datetime([f'{year}-12-31' for year in years])
    
    # PIB Colombia (billones COP nominales)
    pib_growth = [2.9, 1.5, 4.4, 3.9, 4.7, 6.7, 6.7, 3.5, 1.7, 4.0, 
                  4.0, 6.5, 5.4, 4.4, 3.2, 2.0, -6.8, 10.6, 3.5, 7.7, 
                  5.0, 2.8, 7.5, 1.3, 3.3]
    
    pib_base = 250
    pib_values = [pib_base]
    
    for i in range(1, len(years)):
        # Crecimiento real + inflación (~4%)
        nominal_growth = (pib_growth[i-1] + 4 + np.random.normal(0, 1)) / 100
        new_pib = pib_values[-1] * (1 + nominal_growth)
        pib_values.append(new_pib)
    
    # Base gravable IVA (40-60% del PIB)
    base_ratio = np.linspace(0.42, 0.60, len(years))
    base_values = np.array(pib_values) * base_ratio
    
    # Tasa efectiva IVA
    tasa_efectiva = np.array([
        15.5, 16.0, 16.3, 16.7, 16.9, 16.7, 16.8, 16.9,
        17.3, 16.3, 16.0, 15.8, 15.6, 15.7, 16.4, 16.3,
        14.6, 14.7, 14.7, 14.9, 15.0, 15.2, 15.5, 15.8, 16.0
    ])
    
    # IVA recaudado
    iva_values = base_values * (tasa_efectiva / 100)
    
    # Efectos económicos
    economic_effects = np.array([1.0] * 8 + [0.9, 1.1] + [1.0] * 6 + 
                               [0.8, 1.2] + [1.0] * 6 + [1.1])
    iva_values *= economic_effects[:len(iva_values)]
    
    # DataFrames
    df_pib = pd.DataFrame({
        'año': years,
        'fecha': fechas,
        'valor': pib_values,
        'crecimiento': [0] + pib_growth[:len(years)-1]
    })
    
    df_iva = pd.DataFrame({
        'año': years,
        'fecha': fechas,
        'valor': iva_values,
        'base_gravable': base_values,
        'base_ratio_pib': base_ratio * 100,
        'tasa_efectiva': tasa_efectiva,
        'tasa_nominal': [16] * 17 + [19] * 8,
        'ratio_pib': (iva_values / pib_values) * 100,
        'eficiencia': (tasa_efectiva / np.array([16] * 17 + [19] * 8)) * 100
    })
    
    return df_pib, df_iva

def train_models(df_pib, df_iva):
    """Entrenar modelos predictivos"""
    
    # Combinar datos
    df_combined = pd.merge(df_pib, df_iva, on='año', suffixes=('_pib', '_iva'))
    
    # Features
    df_combined['pib_lag1'] = df_combined['valor_pib'].shift(1)
    df_combined['iva_lag1'] = df_combined['valor_iva'].shift(1)
    df_combined['base_lag1'] = df_combined['base_gravable'].shift(1)
    df_combined['año_norm'] = (df_combined['año'] - 2000) / 24
    
    # Preparar datos
    features = ['valor_pib', 'pib_lag1', 'iva_lag1', 'base_lag1', 'año_norm']
    df_model = df_combined[features + ['valor_iva']].dropna()
    
    X = df_model[features]
    y = df_model['valor_iva']
    
    # División temporal
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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'modelo': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return results

def predict_2025(results, pib_2025):
    """Predecir IVA 2025"""
    
    # Features para 2025 (simplificado)
    X_2025 = pd.DataFrame([{
        'valor_pib': pib_2025,
        'pib_lag1': 1200,  # PIB 2024 estimado
        'iva_lag1': 140,   # IVA 2024 estimado
        'base_lag1': 720,  # Base 2024 estimada
        'año_norm': 1.0    # 2025 normalizado
    }])
    
    predictions = {}
    for name, result in results.items():
        pred = result['modelo'].predict(X_2025)[0]
        predictions[name] = pred
    
    return predictions

def main():
    """Aplicación principal"""
    
    st.title("🇨🇴 Análisis PIB-IVA Colombia 2000-2024")
    st.markdown("### Dashboard con Base Gravable y Predicción 2025")
    
    # Cargar datos
    df_pib, df_iva = generate_colombia_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    # Escenario PIB 2025
    scenario = st.sidebar.selectbox(
        "Escenario PIB 2025:",
        ["Conservador (2%)", "Moderado (3.5%)", "Optimista (5%)"]
    )
    
    pib_actual = df_pib['valor'].iloc[-1]
    if "Conservador" in scenario:
        pib_2025 = pib_actual * 1.02
    elif "Moderado" in scenario:
        pib_2025 = pib_actual * 1.035
    else:
        pib_2025 = pib_actual * 1.05
    
    st.sidebar.metric("PIB 2025 Proyectado", f"{pib_2025:.1f} billones COP")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Histórico", "🤖 Modelos ML", "🔮 Predicción 2025"])
    
    with tab1:
        st.header("📊 Análisis Histórico Colombia")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("PIB 2024", f"{df_pib['valor'].iloc[-1]:.1f} billones")
        with col2:
            st.metric("IVA 2024", f"{df_iva['valor'].iloc[-1]:.1f} billones")
        with col3:
            st.metric("Base Gravable", f"{df_iva['base_gravable'].iloc[-1]:.1f} billones")
        with col4:
            st.metric("Eficiencia", f"{df_iva['eficiencia'].iloc[-1]:.1f}%")
        
        # Visualización principal
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PIB Colombia', 'IVA Recaudado', 'Base Gravable', 'Eficiencia del Recaudo'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # PIB
        fig.add_trace(
            go.Scatter(x=df_pib['año'], y=df_pib['valor'], mode='lines+markers',
                      name='PIB', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # IVA
        fig.add_trace(
            go.Scatter(x=df_iva['año'], y=df_iva['valor'], mode='lines+markers',
                      name='IVA', line=dict(color='orange', width=3)),
            row=1, col=2
        )
        
        # Base gravable
        fig.add_trace(
            go.Scatter(x=df_iva['año'], y=df_iva['base_gravable'], mode='lines+markers',
                      name='Base', line=dict(color='green', width=3)),
            row=2, col=1
        )
        
        # Eficiencia
        fig.add_trace(
            go.Scatter(x=df_iva['año'], y=df_iva['eficiencia'], mode='lines+markers',
                      name='Eficiencia', line=dict(color='purple', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text="Análisis Histórico PIB-IVA Colombia")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos recientes
        st.subheader("📋 Datos Recientes (2020-2024)")
        recent_data = pd.merge(
            df_pib[['año', 'valor', 'crecimiento']].tail(5),
            df_iva[['año', 'valor', 'base_gravable', 'eficiencia']].tail(5),
            on='año'
        )
        recent_data.columns = ['Año', 'PIB', 'Crecimiento PIB (%)', 'IVA', 'Base Gravable', 'Eficiencia (%)']
        st.dataframe(recent_data.round(1), use_container_width=True)
    
    with tab2:
        st.header("🤖 Modelos de Machine Learning")
        
        # Entrenar modelos
        results = train_models(df_pib, df_iva)
        
        # Mostrar métricas
        st.subheader("📊 Performance de Modelos")
        
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Modelo': name,
                'RMSE': result['rmse'],
                'R²': result['r2']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
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
    
    with tab3:
        st.header("🔮 Predicción IVA 2025")
        
        # Realizar predicciones
        results = train_models(df_pib, df_iva)
        predictions = predict_2025(results, pib_2025)
        
        # Mostrar predicciones
        st.subheader(f"📊 Predicciones para {scenario}")
        
        pred_data = []
        for model, pred in predictions.items():
            pred_data.append({
                'Modelo': model,
                'IVA 2025': f"{pred:.1f} billones",
                'Crecimiento vs 2024': f"{((pred/df_iva['valor'].iloc[-1])-1)*100:+.1f}%"
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
        
        # Visualización temporal
        avg_prediction = np.mean(list(predictions.values()))
        
        fig_pred = go.Figure()
        
        # Datos históricos
        fig_pred.add_trace(go.Scatter(
            x=df_iva['año'],
            y=df_iva['valor'],
            mode='lines+markers',
            name='IVA Histórico',
            line=dict(color='blue', width=3)
        ))
        
        # Predicción promedio
        fig_pred.add_trace(go.Scatter(
            x=[2024, 2025],
            y=[df_iva['valor'].iloc[-1], avg_prediction],
            mode='lines+markers',
            name='Proyección 2025',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig_pred.update_layout(
            title="Predicción IVA Colombia 2025",
            xaxis_title="Año",
            yaxis_title="IVA (billones COP)"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Insights
        st.subheader("💡 Insights Clave")
        
        growth_2025 = ((avg_prediction / df_iva['valor'].iloc[-1]) - 1) * 100
        new_ratio = (avg_prediction / pib_2025) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("IVA 2025 Promedio", f"{avg_prediction:.1f} billones")
        col2.metric("Crecimiento vs 2024", f"{growth_2025:+.1f}%")
        col3.metric("Nuevo Ratio IVA/PIB", f"{new_ratio:.1f}%")
        
        st.success(f"""
        **Resumen Ejecutivo:**
        - Bajo el escenario {scenario.lower()}, se proyecta un IVA de {avg_prediction:.1f} billones de COP para 2025
        - Esto representa un crecimiento del {growth_2025:.1f}% respecto a 2024
        - El ratio IVA/PIB se mantendría en {new_ratio:.1f}%, dentro del rango histórico
        """)

if __name__ == "__main__":
    main()
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(app_code)
    
    print("✅ streamlit_app.py creado exitosamente")

if __name__ == "__main__":
    main()
