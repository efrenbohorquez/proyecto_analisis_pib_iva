@echo off
title Verificar Dashboard PIB-IVA Colombia

echo.
echo ==========================================
echo 🔍 VERIFICACION DASHBOARD PIB-IVA COLOMBIA
echo ==========================================
echo 📊 Revisando cambios implementados
echo 🚀 Preparando ejecución
echo ==========================================
echo.

cd /d "%~dp0"
python verificar_y_ejecutar.py

echo.
echo ==========================================
echo ✅ Verificación completada
echo ==========================================
pause
