# 🚀 Guía de Ejecución - Dashboard PIB-IVA Colombia

## ⚡ Opciones de Ejecución

### 1. Inicio Automático (Recomendado)
```cmd
# Doble clic en cualquiera de estos archivos:
EJECUTAR_DASHBOARD.bat    # Versión completa con verificaciones
quick_start.bat          # Versión rápida
```

### 2. Ejecución con Python
```cmd
cd d:\Downloads\proyecto_analisis_pib_iva
python launch_dashboard.py
```

### 3. Streamlit Directo
```cmd
cd d:\Downloads\proyecto_analisis_pib_iva
streamlit run streamlit_app.py
```

## 📊 Funcionalidades del Dashboard

### 🔍 **Análisis Disponibles:**
- ✅ **PIB-IVA Histórico** (2000-2024)
- ✅ **Transformaciones Box-Cox** para estabilizar varianza
- ✅ **Modelos ARIMA/SARIMA** con selección automática
- ✅ **SARIMAX** con PIB como variable exógena
- ✅ **Modelos VAR** multivariados
- ✅ **Machine Learning** (Random Forest, Gradient Boosting)
- ✅ **Predicción Consenso 2025** combinando todos los modelos

### 📈 **6 Tabs Principales:**
1. **📊 Análisis Detallado** - Métricas y datos históricos
2. **📈 Box-Cox & Transformaciones** - Estabilización de varianza
3. **🔄 ARIMA & SARIMA** - Modelos univariados
4. **🌐 SARIMAX & VAR** - Modelos con variables exógenas
5. **🤖 Machine Learning** - Algoritmos de aprendizaje automático
6. **🔮 Predicción 2025** - Consenso de todos los modelos

## 🔧 Requisitos Técnicos

### **Software Necesario:**
- Python 3.8+ 
- 8GB RAM mínimo
- Conexión a internet (primera ejecución)

### **Dependencias Instaladas Automáticamente:**
```
streamlit plotly pandas numpy scikit-learn
statsmodels pmdarima arch scipy
```

## 🌐 Acceso al Dashboard

- **URL Local**: http://localhost:8501
- **Puerto**: 8501
- **Auto-apertura**: Sí (navegador predeterminado)

## ❓ Solución de Problemas

### **Error: "Python no encontrado"**
```cmd
# Instalar Python desde: https://python.org
# Asegurar que esté en PATH del sistema
```

### **Error: "Puerto en uso"**
```cmd
# Cambiar puerto:
streamlit run streamlit_app.py --server.port 8502
```

### **Error: "Dependencias faltantes"**
```cmd
# Instalar manualmente:
pip install streamlit plotly pandas numpy
pip install statsmodels pmdarima arch
```

### **Error: "Archivo no encontrado"**
```cmd
# Verificar ubicación:
cd d:\Downloads\proyecto_analisis_pib_iva
dir *.py
```

## 📱 Uso del Dashboard

1. **Ejecutar** cualquier script .bat
2. **Esperar** instalación automática de dependencias  
3. **Abrir** http://localhost:8501 en navegador
4. **Explorar** las 6 pestañas de análisis
5. **Configurar** escenarios en sidebar
6. **Generar** predicciones 2025

## 🎯 Casos de Uso

- **Análisis macroeconómico** de Colombia
- **Proyección fiscal** del IVA
- **Modelado econométrico** avanzado
- **Investigación académica** en economía
- **Toma de decisiones** de política tributaria

¡El dashboard está listo para análisis profesional de series temporales económicas!
