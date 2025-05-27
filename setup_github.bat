@echo off
title Configurar GitHub - PIB-IVA Colombia

echo.
echo =========================================
echo 🔗 CONFIGURAR REPOSITORIO GITHUB
echo =========================================
echo 📂 Proyecto: PIB-IVA Colombia 2000-2024
echo =========================================
echo.

cd /d "d:\Downloads\proyecto_analisis_pib_iva"

echo 🔍 Verificando Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git no encontrado
    echo 📥 Instala Git desde: https://git-scm.com
    pause
    exit /b 1
)

echo ✅ Git disponible
echo.

echo 📁 Inicializando repositorio local...
git init

echo 📝 Configurando archivos...
echo node_modules/ > .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
echo .env >> .gitignore
echo .streamlit/secrets.toml >> .gitignore

echo 📊 Agregando archivos al repositorio...
git add .
git commit -m "Dashboard PIB-IVA Colombia con análisis de series temporales

- Análisis histórico PIB-IVA Colombia 2000-2024
- Modelos ARIMA, SARIMA, SARIMAX, VAR
- Transformaciones Box-Cox
- Machine Learning comparativo
- Predicción consenso IVA 2025
- Dashboard Streamlit interactivo"

echo.
echo 🌐 OPCIONES DE CONFIGURACIÓN REMOTA:
echo.
echo 1. Crear nuevo repositorio en GitHub:
echo    https://github.com/new
echo.
echo 2. Vincular repositorio existente:
echo    git remote add origin https://github.com/TU_USUARIO/proyecto-analisis-pib-iva.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. Para Streamlit Cloud:
echo    - Crear cuenta en https://streamlit.io
echo    - Conectar repositorio GitHub
echo    - Desplegar automáticamente
echo.

set /p respuesta="¿Configurar repositorio remoto ahora? (s/n): "
if /i "%respuesta%"=="s" (
    set /p repo_url="Ingresa URL del repositorio GitHub: "
    git remote add origin !repo_url!
    git branch -M main
    echo 📤 Subiendo a GitHub...
    git push -u origin main
    echo ✅ Repositorio configurado exitosamente
) else (
    echo 💡 Configuración manual disponible arriba
)

echo.
echo ✅ Configuración completada
pause
