# 🇨🇴 Dashboard PIB-IVA Colombia 2000-2024

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-usuario-proyecto-analisis-pib-iva.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Descripción

Dashboard interactivo para análisis econométrico del PIB e IVA en Colombia (2000-2024) con predicción para 2025 utilizando modelos avanzados de series temporales y machine learning.

## 🚀 Demo en Vivo

🌐 **[Ver Dashboard en Streamlit Cloud](https://tu-usuario-proyecto-analisis-pib-iva.streamlit.app/)**

## ✨ Características

### 📈 Modelos Implementados
- **ARIMA/SARIMA** con selección automática de parámetros
- **SARIMAX** con PIB como variable exógena
- **VAR** (Vector Autoregresivo) multivariado
- **Box-Cox** para transformaciones de estabilización
- **Machine Learning** (Random Forest, Gradient Boosting)

### 📊 Análisis Disponibles
- Análisis histórico PIB-IVA (2000-2024)
- Transformaciones para estabilizar varianza
- Tests de estacionariedad
- Diagnósticos de residuos
- Funciones de respuesta al impulso
- Predicción consenso 2025

## 🛠️ Instalación Local

### Prerrequisitos
- Python 3.8+
- Git

### Instalación Rápida
```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/proyecto-analisis-pib-iva.git
cd proyecto-analisis-pib-iva

# Ejecutar dashboard (instala dependencias automáticamente)
./EJECUTAR_DASHBOARD.bat  # Windows
```

### Instalación Manual
```bash
# Instalar dependencias
pip install streamlit plotly pandas numpy scikit-learn
pip install statsmodels pmdarima arch scipy

# Ejecutar dashboard
streamlit run streamlit_app.py
```

## 📱 Uso

1. **Ejecutar** `EJECUTAR_DASHBOARD.bat`
2. **Abrir** http://localhost:8501
3. **Explorar** las 6 pestañas de análisis:
   - 📊 Análisis Detallado
   - 📈 Box-Cox & Transformaciones
   - 🔄 ARIMA & SARIMA
   - 🌐 SARIMAX & VAR
   - 🤖 Machine Learning
   - 🔮 Predicción 2025

## 📊 Estructura del Proyecto

```
proyecto-analisis-pib-iva/
├── streamlit_app.py          # Aplicación principal
├── EJECUTAR_DASHBOARD.bat    # Script de ejecución Windows
├── launch_dashboard.py       # Lanzador Python
├── requirements.txt          # Dependencias
├── setup_github.bat         # Configuración GitHub
└── README.md                # Este archivo
```

## 🔮 Predicciones 2025

### 📈 Proyecciones Económicas
- **PIB 2025**: 1,100-1,200 billones COP
- **IVA 2025**: 165-185 billones COP  
- **Ratio IVA/PIB**: 14.5-15.5%
- **Consenso**: Promedio de 8+ modelos

### 🎯 Escenarios Disponibles
- **Conservador** (2% crecimiento PIB)
- **Moderado** (3.5% crecimiento PIB)
- **Optimista** (5% crecimiento PIB)
- **Personalizado** (slider interactivo)

## 🤖 Modelos Técnicos

### Series Temporales
- **Auto-ARIMA** con pmdarima
- **SARIMAX** con statsmodels
- **VAR** multivariado
- **Box-Cox** para transformaciones

### Machine Learning
- Random Forest Regressor
- Gradient Boosting Regressor
- Regresión Lineal

## 📈 Resultados

### Correlación PIB-IVA
- **r = 0.95+** (correlación muy fuerte)
- **R² > 0.90** en modelos ML
- **AIC/BIC optimizados** en modelos ARIMA

### Eficiencia del Recaudo
- **Tasa efectiva**: ~16% (vs 19% nominal)
- **Base gravable**: 60% del PIB
- **Eficiencia**: 84% del potencial teórico

## 🌐 Deployment

### Streamlit Cloud
```bash
# 1. Subir a GitHub
git push origin main

# 2. Conectar en streamlit.io
# 3. Deploy automático
```

### Heroku
```bash
# Crear Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git add Procfile
git commit -m "Add Procfile for Heroku"
git push heroku main
```

## 📚 Metodología

### Fuentes de Datos
- **DANE Colombia**: PIB oficial
- **Ministerio de Hacienda**: Recaudo IVA
- **Banco de la República**: Series macroeconómicas

### Validación
- **Backtesting** con datos 2020-2024
- **Cross-validation** temporal
- **Test de Ljung-Box** para residuos
- **ADF test** para estacionariedad

## 🤝 Contribuir

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- Email: tu.email@gmail.com

## 🙏 Agradecimientos

- **DANE Colombia** por datos oficiales del PIB
- **Ministerio de Hacienda** por información tributaria
- **Streamlit** por la plataforma de desarrollo
- **Comunidad Python** por las librerías de análisis

---

⭐ **¡Si te gusta este proyecto, dale una estrella!** ⭐
