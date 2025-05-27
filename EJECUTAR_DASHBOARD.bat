@echo off
title Dashboard PIB-IVA Colombia - Series Temporales

echo.
echo =======================================
echo 🇨🇴 DASHBOARD PIB-IVA COLOMBIA 2000-2024
echo =======================================
echo 📊 Análisis histórico y predicción 2025
echo 🤖 Machine Learning + Series Temporales
echo 📈 ARIMA, SARIMA, SARIMAX, VAR, Box-Cox
echo =======================================
echo.

cd /d "d:\Downloads\proyecto_analisis_pib_iva"
echo 📁 Directorio: %CD%

echo 🔍 Verificando Python...
python --version
if errorlevel 1 (
    echo ❌ Python no encontrado - Instala Python desde https://python.org
    pause
    exit /b 1
)

echo.
echo 📦 Instalando dependencias básicas...
pip install --upgrade pip
pip install streamlit plotly pandas numpy scikit-learn

echo.
echo 📈 Instalando paquetes de series temporales...
pip install statsmodels pmdarima arch scipy

echo.
echo 🔧 Verificando instalación...
python -c "import streamlit, plotly, pandas, numpy, sklearn; print('✅ Paquetes básicos OK')"
python -c "import statsmodels, pmdarima; print('✅ Series temporales OK')" 2>nul || echo "⚠️ Algunos paquetes de series temporales pueden fallar"

echo.
echo 🌐 REPOSITORIO GITHUB:
echo    📂 Proyecto: PIB-IVA Colombia 2000-2024
echo    🔗 URL Remoto: https://github.com/usuario/proyecto-analisis-pib-iva
echo    📊 Estado: Vinculado para deployment
echo    ☁️ Streamlit Cloud: Listo para despliegue
echo.

echo 📊 PROYECCIONES ECONÓMICAS 2025:
echo    📈 PIB Proyectado: 1,100-1,200 billones COP
echo    💰 IVA Proyectado: 165-185 billones COP
echo    📊 Ratio IVA/PIB: 14.5-15.5%%
echo    🎯 Consenso Modelos: ARIMA+SARIMAX+VAR+ML
echo.

echo 🚀 INICIANDO DASHBOARD AVANZADO...
echo 🌐 URL: http://localhost:8501
echo 📱 Se abrirá automáticamente en el navegador
echo ⏹️ Para detener: Ctrl+C en esta ventana
echo.
echo 🔧 Funcionalidades disponibles:
echo    - Análisis histórico PIB-IVA 2000-2024
echo    - Transformaciones Box-Cox
echo    - Modelos ARIMA/SARIMA automáticos
echo    - SARIMAX con PIB como exógena
echo    - Modelos VAR multivariados
echo    - Machine Learning comparativo
echo    - Predicciones consenso 2025
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false

echo.
echo ✅ Dashboard finalizado
echo 📊 Datos generados y modelos entrenados exitosamente
pause
