# 📚 Manual de Usuario - Proyecto Análisis PIB-IVA

## 🎯 Introducción

Este proyecto implementa un sistema moderno de análisis predictivo para la relación entre PIB e IVA en España, diseñado específicamente para cumplir con estándares académicos de maestría en ciencia de datos.

## 🏗️ Arquitectura del Sistema

### Estructura del Proyecto
```
proyecto_analisis_pib_iva/
├── config/                     # Configuración centralizada
│   └── config.py               # Configuraciones del proyecto
├── src/                        # Código fuente principal
│   ├── data/                   # Módulos de datos
│   │   ├── data_validator.py   # Validación de datos
│   │   └── preprocessor.py     # Preprocesamiento
│   ├── models/                 # Modelos de ML
│   │   ├── base_model.py       # Clase base de modelos
│   │   └── ensemble_model.py   # Modelos ensemble
│   ├── visualization/          # Visualizaciones
│   │   └── academic_plots.py   # Gráficos académicos
│   └── utils/                  # Utilidades
│       └── logger.py           # Sistema de logging
├── data/                       # Datos del proyecto
├── models/                     # Modelos entrenados
├── output/                     # Resultados y visualizaciones
├── logs/                       # Archivos de log
└── docs/                       # Documentación
```

## 🚀 Instalación y Configuración

### Requisitos del Sistema
- Python 3.8+
- 8GB RAM mínimo (16GB recomendado)
- 2GB espacio en disco

### Instalación de Dependencias
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración Inicial
```python
from config.config import config

# Verificar configuración
print(f"Directorio de datos: {config.data_dir}")
print(f"Directorio de salida: {config.output_dir}")
```

## 📊 Uso del Sistema

### 1. Análisis Exploratorio de Datos

```python
from src.data.data_validator import DataValidator
from src.visualization.academic_plots import AcademicPlotGenerator

# Cargar datos
df_pib = pd.read_csv("data/pib_historico.csv")
df_iva = pd.read_csv("data/iva_historico.csv")

# Validar datos
validator = DataValidator()
validation_pib = validator.validate_pib_data(df_pib)
validation_iva = validator.validate_iva_data(df_iva)

# Crear visualizaciones académicas
plotter = AcademicPlotGenerator()
fig_temporal = plotter.create_time_series_analysis(df_pib, df_iva)
fig_correlation = plotter.create_correlation_analysis(df_pib, df_iva)
```

### 2. Entrenamiento de Modelos

```python
from src.models.ensemble_model import EnsemblePredictor

# Inicializar predictor
predictor = EnsemblePredictor()

# Entrenar múltiples modelos
results = predictor.train_multiple_models(df_pib, df_iva)

# Obtener mejor modelo
best_model = predictor.get_best_model()
```

### 3. Generación de Reportes

```python
from src.reporting.academic_report import AcademicReportGenerator

# Generar reporte completo
report_generator = AcademicReportGenerator()
report_generator.generate_full_report(
    pib_data=df_pib,
    iva_data=df_iva,
    model_results=results,
    output_path="output/reporte_completo.html"
)
```

## 📈 Interpretación de Resultados

### Métricas de Evaluación

| Métrica | Descripción | Rango Óptimo |
|---------|-------------|--------------|
| R² | Coeficiente de determinación | 0.85 - 0.95 |
| RMSE | Error cuadrático medio | < 5% de la media |
| MAE | Error absoluto medio | < 3% de la media |
| MAPE | Error porcentual absoluto medio | < 5% |

### Interpretación de Visualizaciones

#### Análisis Temporal
- **Tendencia**: Patrón de largo plazo
- **Estacionalidad**: Patrones recurrentes
- **Ciclos**: Fluctuaciones económicas

#### Análisis de Correlación
- **Correlación cruzada**: Relación temporal entre variables
- **Lag óptimo**: Retraso temporal de máxima correlación
- **Estabilidad**: Consistencia de la relación

## 🔧 Configuración Avanzada

### Personalización de Modelos

```python
# Configurar hiperparámetros
config.models.random_forest = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 3
}

# Configurar validación cruzada
config.models.cross_validation_folds = 10
```

### Personalización de Visualizaciones

```python
# Configurar tema académico
config.visualization.theme = "plotly_white"
config.visualization.color_primary = "#2E86AB"
config.visualization.font_size = 16
```

## 🐛 Resolución de Problemas

### Errores Comunes

#### Error: "Datos insuficientes"
**Solución**: Verificar que los datasets tengan al menos 24 observaciones.

#### Error: "Correlación muy baja"
**Solución**: Revisar la calidad de los datos y considerar transformaciones.

#### Error: "Modelo no converge"
**Solución**: Ajustar hiperparámetros o cambiar algoritmo.

### Diagnóstico de Performance

```python
# Verificar calidad de datos
validation_result = validator.validate_relationship(df_pib, df_iva)
print(f"Correlación: {validation_result.summary['correlation']:.3f}")

# Analizar residuos del modelo
residuals_analysis = predictor.analyze_residuals()
```

## 📝 Buenas Prácticas

### Preparación de Datos
1. **Validar** siempre los datos antes del análisis
2. **Documentar** transformaciones aplicadas
3. **Verificar** consistencia temporal
4. **Detectar** y tratar outliers apropiadamente

### Modelado
1. **Dividir** datos temporalmente (no aleatoriamente)
2. **Validar** con datos fuera de muestra
3. **Comparar** múltiples algoritmos
4. **Documentar** supuestos del modelo

### Visualización
1. **Usar** escalas apropiadas
2. **Incluir** intervalos de confianza
3. **Añadir** contexto económico
4. **Seguir** principios de diseño académico

## 🎓 Estándares Académicos

### Documentación Requerida
- [ ] Descripción metodológica completa
- [ ] Justificación de algoritmos seleccionados
- [ ] Análisis de limitaciones
- [ ] Validación estadística rigurosa
- [ ] Interpretación económica de resultados

### Visualizaciones Académicas
- [ ] Títulos descriptivos y específicos
- [ ] Ejes claramente etiquetados
- [ ] Leyendas informativas
- [ ] Fuentes de datos citadas
- [ ] Calidad de impresión (300 DPI)

## 📞 Soporte

### Logs del Sistema
```bash
# Verificar logs de errores
tail -f logs/DataValidator_20231201.log

# Logs de modelos
tail -f logs/EnsemblePredictor_20231201.log
```

### Contacto
- **Documentación**: `docs/`
- **Issues**: Crear ticket en repositorio
- **Logs**: Revisar `logs/` para diagnóstico
