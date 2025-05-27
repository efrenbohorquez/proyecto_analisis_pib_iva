# 📊 Estructura del Dashboard de Predicción de IVA

## 🔄 Orden de Ejecución de Visualizaciones

El dashboard se ejecuta en el siguiente orden secuencial:

### 1. **📈 Series Temporales Principales** (`grafico_series_temporales`)
- **Ubicación**: Primera visualización
- **Contenido**: 
  - Gráfico superior: Evolución del PIB con media móvil (12 meses)
  - Gráfico inferior: Evolución del IVA con media móvil (12 meses)
- **Interactividad**: Hover con valores exactos, zoom, pan
- **Objetivo**: Mostrar tendencias históricas de ambas variables

### 2. **🔗 Análisis de Correlación Avanzado** (`analisis_correlacion_avanzado`)
- **Ubicación**: Segunda visualización
- **Contenido**:
  - Panel izquierdo: Correlación PIB-IVA por lag (-6 a +6 meses)
  - Panel derecho: Scatter plot PIB vs IVA con línea de tendencia
- **Métricas mostradas**: Correlación máxima y lag óptimo
- **Objetivo**: Identificar relaciones temporales entre variables

### 3. **🎯 Evaluación del Modelo** (`visualizar_performance_modelo`)
- **Ubicación**: Tercera visualización
- **Contenido** (4 paneles):
  - Superior izquierdo: Predicciones vs Valores Reales
  - Superior derecho: Distribución de Residuos (histograma)
  - Inferior izquierdo: Residuos vs Predicciones
  - Inferior derecho: Residuos en el Tiempo
- **Objetivo**: Evaluar calidad del modelo de predicción

### 4. **🔮 Predicciones Futuras** (`grafico_predicciones_futuras`)
- **Ubicación**: Cuarta visualización
- **Contenido**:
  - Serie histórica del IVA
  - Predicciones futuras (6 meses)
  - Intervalo de confianza
  - Línea divisoria entre histórico y predicciones
- **Objetivo**: Mostrar proyecciones del modelo

### 5. **📊 Dashboard de Métricas Interactivo** (`dashboard_metricas_interactivo`)
- **Ubicación**: Quinta visualización
- **Contenido** (4 gauges):
  - RMSE (superior izquierdo)
  - MAE (superior derecho)
  - MAPE (inferior izquierdo)
  - R² (inferior derecho)
- **Semáforos**: Verde (excelente), Amarillo (bueno), Rojo (mejorable)
- **Objetivo**: Resumen visual de performance del modelo

### 6. **🔄 Análisis de Estacionalidad** (`analisis_estacionalidad_moderno`)
- **Ubicación**: Sexta y última visualización
- **Contenido** (4 paneles):
  - Superior izquierdo: Patrón Estacional Mensual (barras)
  - Superior derecho: Patrón Estacional Trimestral (barras)
  - Inferior izquierdo: Descomposición de Tendencia y Componente Estacional
  - Inferior derecho: Evolución Anual
- **Objetivo**: Identificar patrones temporales en el IVA

## 🎨 Características de Diseño

### Esquema de Colores Consistente:
- **PIB**: `#1f77b4` (Azul)
- **IVA**: `#ff7f0e` (Naranja) 
- **Predicciones**: `#2ca02c` (Verde)
- **Residuos/Errores**: `#d62728` (Rojo)

### Elementos Interactivos:
- ✅ Hover tooltips personalizados
- ✅ Zoom y pan en gráficos temporales
- ✅ Hover unificado en series temporales
- ✅ Marcadores específicos para predicciones

### Template Visual:
- **Base**: `plotly_white` para apariencia limpia
- **Fuentes**: Tamaño 12px para legibilidad
- **Líneas**: Grosor 3px para series principales, 2px para secundarias

## 📱 Responsividad y Dimensiones

| Visualización | Alto (px) | Ancho | Especial |
|---------------|-----------|-------|----------|
| Series Temporales | 600 | Auto | 2 subplots verticales |
| Correlación | 500 | Auto | 2 subplots horizontales |
| Performance Modelo | 700 | Auto | 4 subplots (2x2) |
| Predicciones Futuras | 500 | Auto | Serie única con áreas |
| Métricas (Gauges) | 600 | Auto | 4 indicadores (2x2) |
| Estacionalidad | 700 | Auto | 4 subplots (2x2) |

## 🔧 Configuración Técnica

### Librerías Utilizadas:
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
```

### Parámetros Clave:
- **Ventana MA**: 12 meses para medias móviles
- **Lags analizados**: -6 a +6 meses
- **Bins histograma**: 20 para residuos
- **Horizonte predicción**: 6 meses
- **Intervalos gauge**: Verde (0-umbral1), Amarillo (umbral1-umbral2), Rojo (umbral2+)
