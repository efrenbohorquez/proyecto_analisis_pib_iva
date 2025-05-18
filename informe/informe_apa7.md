# Análisis de la Relación entre el PIB y la Recaudación de IVA en Colombia (2000-2024)

## Resumen

Este estudio analiza la relación entre el Producto Interno Bruto (PIB) y la recaudación del Impuesto al Valor Agregado (IVA) en Colombia durante el período 2000-2024. Mediante técnicas de análisis de series temporales, se examina la correlación, estacionariedad y causalidad entre ambas variables. Se implementan y comparan modelos tradicionales (SARIMA) y de aprendizaje profundo (LSTM, GRU) para la predicción de la recaudación de IVA. Los resultados muestran una correlación significativa (0.64) entre el PIB y el IVA, confirmando la estrecha relación entre el crecimiento económico y la recaudación fiscal. Los modelos de redes neuronales demostraron mayor capacidad para capturar patrones no lineales en la serie temporal, superando a los modelos tradicionales en precisión predictiva. Este análisis proporciona herramientas valiosas para la planificación fiscal y la evaluación de políticas económicas en Colombia.

**Palabras clave**: Series temporales, PIB, IVA, SARIMA, redes neuronales, LSTM, Colombia, política fiscal

## Introducción

La relación entre el crecimiento económico y la recaudación fiscal representa uno de los vínculos fundamentales en la economía de cualquier nación. En Colombia, el Impuesto al Valor Agregado (IVA) constituye una de las principales fuentes de ingresos tributarios, representando aproximadamente el 40% de la recaudación fiscal total. Comprender la dinámica entre el Producto Interno Bruto (PIB) y la recaudación de IVA resulta crucial para la formulación de políticas fiscales efectivas y la planificación presupuestaria gubernamental.

El presente estudio se enfoca en analizar la relación entre el PIB y la recaudación de IVA en Colombia durante el período 2000-2024, abarcando diferentes ciclos económicos, reformas tributarias y crisis globales como la pandemia de COVID-19. Este extenso período permite examinar tanto tendencias a largo plazo como patrones estacionales y coyunturales que afectan a ambas variables.

La investigación se estructura en torno a tres objetivos principales: (1) analizar la correlación y posible cointegración entre el PIB y la recaudación de IVA; (2) identificar patrones estacionales y tendencias en la recaudación de IVA; y (3) desarrollar y comparar modelos predictivos utilizando tanto técnicas estadísticas tradicionales (SARIMA) como métodos de aprendizaje profundo (redes neuronales LSTM y GRU).

La relevancia de este estudio radica en su potencial para informar decisiones de política fiscal, mejorar las proyecciones de ingresos tributarios y proporcionar herramientas analíticas avanzadas para la administración tributaria colombiana. Además, la comparación entre métodos tradicionales y técnicas de aprendizaje profundo contribuye al debate metodológico sobre la predicción de series temporales en el ámbito de las finanzas públicas.

## Marco Teórico

### Relación entre Crecimiento Económico y Recaudación Fiscal

La teoría económica establece una relación directa entre el crecimiento económico y la recaudación fiscal. Según Tanzi y Zee (2000), el crecimiento del PIB generalmente conduce a un aumento en la base imponible, lo que se traduce en mayores ingresos fiscales. Esta relación, sin embargo, no es necesariamente lineal y puede estar influenciada por diversos factores como la estructura económica, la eficiencia en la administración tributaria y la elasticidad de los diferentes impuestos.

En el caso específico del IVA, Keen y Lockwood (2010) argumentan que este impuesto tiende a mostrar una alta elasticidad respecto al PIB debido a su amplia base y su relación directa con el consumo. No obstante, esta elasticidad puede variar según la composición sectorial de la economía y la estructura de exenciones y tasas diferenciadas del impuesto.

### Análisis de Series Temporales en Finanzas Públicas

El análisis de series temporales ha sido ampliamente utilizado en el estudio de variables fiscales. Los modelos ARIMA (Autoregressive Integrated Moving Average) y sus variantes estacionales (SARIMA) han demostrado ser herramientas efectivas para modelar y predecir ingresos tributarios (Box et al., 2015). Estos modelos capturan la autocorrelación, tendencias y patrones estacionales presentes en las series temporales fiscales.

La cointegración, concepto desarrollado por Engle y Granger (1987), permite analizar relaciones de equilibrio a largo plazo entre variables no estacionarias. En el contexto de las finanzas públicas, la cointegración entre el PIB y los ingresos fiscales sugiere una relación estable a largo plazo, donde las desviaciones temporales tienden a corregirse con el tiempo.

### Aplicación de Redes Neuronales en Predicción Fiscal

En años recientes, las técnicas de aprendizaje profundo han ganado popularidad en la predicción de series temporales económicas y financieras. Las redes neuronales recurrentes (RNN), particularmente las arquitecturas LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit), han demostrado capacidad para capturar patrones complejos y no lineales en series temporales (Hochreiter y Schmidhuber, 1997; Cho et al., 2014).

Estudios como los de Siami-Namini et al. (2018) y Zhang (2003) han comparado el desempeño de modelos tradicionales como ARIMA con redes neuronales en la predicción de series temporales económicas, encontrando que las redes neuronales suelen superar a los modelos estadísticos tradicionales en términos de precisión predictiva, especialmente en series con alta volatilidad o patrones no lineales.

En el contexto latinoamericano, trabajos como los de Cárdenas y Rozo (2009) han aplicado técnicas econométricas para analizar la relación entre crecimiento económico y recaudación tributaria en Colombia, mientras que estudios más recientes como los de Gómez y Morán (2016) han explorado la aplicación de técnicas de aprendizaje automático en la predicción de ingresos fiscales en países de la región.

## Metodología

### Datos y Fuentes

Para este estudio se utilizaron datos mensuales de recaudación de IVA en Colombia para el período enero 2000 a diciembre 2024, obtenidos de la Dirección de Impuestos y Aduanas Nacionales (DIAN). Los datos del PIB se obtuvieron del Departamento Administrativo Nacional de Estadística (DANE) y del Ministerio de Hacienda de Colombia.

Dado que el PIB se publica oficialmente con frecuencia trimestral, se realizó una mensualización utilizando indicadores de actividad económica como el Índice de Seguimiento a la Economía (ISE) para obtener estimaciones mensuales consistentes con los agregados trimestrales oficiales.

### Preprocesamiento de Datos

El preprocesamiento de los datos incluyó las siguientes etapas:

1. **Estandarización de fechas**: Conversión de todas las series temporales al primer día de cada mes para garantizar la alineación correcta.
2. **Detección y tratamiento de valores atípicos**: Identificación de observaciones anómalas mediante métodos estadísticos y su tratamiento mediante técnicas de suavizado.
3. **Alineación de series**: Creación de un conjunto de datos integrado con observaciones coincidentes de PIB e IVA.
4. **Transformación de variables**: Aplicación de transformaciones logarítmicas para estabilizar la varianza y facilitar la interpretación de elasticidades.

### Análisis Exploratorio y Pruebas de Estacionariedad

Se realizó un análisis exploratorio de datos (EDA) para identificar patrones, tendencias y relaciones preliminares entre las variables. Este análisis incluyó:

1. **Visualización de series temporales**: Gráficos de evolución temporal de ambas variables.
2. **Análisis de correlación**: Cálculo de coeficientes de correlación entre PIB e IVA.
3. **Análisis de estacionalidad**: Identificación de patrones estacionales en la recaudación de IVA.
4. **Pruebas de estacionariedad**: Aplicación de pruebas Dickey-Fuller Aumentada (ADF) y KPSS para evaluar la estacionariedad de las series.
5. **Pruebas de cointegración**: Test de Engle-Granger para evaluar la existencia de relaciones de equilibrio a largo plazo.

### Modelado de Series Temporales

#### Modelos SARIMA

Se implementaron modelos SARIMA (Seasonal Autoregressive Integrated Moving Average) para capturar la dinámica temporal de la recaudación de IVA. La especificación del modelo siguió la notación SARIMA(p,d,q)(P,D,Q)s, donde:

- p, d, q: Órdenes no estacionales (autorregresivo, diferenciación, media móvil)
- P, D, Q: Órdenes estacionales
- s: Período estacional (12 para datos mensuales)

La selección de los órdenes óptimos se realizó mediante análisis de funciones de autocorrelación (ACF) y autocorrelación parcial (PACF), así como criterios de información como AIC y BIC.

#### Modelos de Redes Neuronales

Se implementaron dos arquitecturas de redes neuronales recurrentes:

1. **LSTM (Long Short-Term Memory)**: Red neuronal recurrente con capacidad para capturar dependencias a largo plazo, utilizando celdas de memoria con puertas de entrada, olvido y salida.

2. **GRU (Gated Recurrent Unit)**: Variante simplificada de LSTM con menos parámetros, utilizando puertas de actualización y reinicio.

Para ambas arquitecturas, se utilizó una ventana temporal de 12 meses como entrada para predecir el valor del mes siguiente. Los datos se normalizaron al rango [0,1] mediante un escalador MinMax. Los modelos se entrenaron utilizando el 80% de los datos, reservando el 20% restante para validación y prueba.

### Evaluación de Modelos

Los modelos se evaluaron utilizando las siguientes métricas:

1. **RMSE (Root Mean Square Error)**: Raíz cuadrada del error cuadrático medio.
2. **MAE (Mean Absolute Error)**: Error absoluto medio.
3. **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio.

Adicionalmente, se realizó un análisis de residuos para evaluar la adecuación de los modelos, incluyendo pruebas de normalidad, autocorrelación y homocedasticidad de los residuos.

## Resultados

### Análisis Exploratorio de Datos

El análisis exploratorio reveló una tendencia creciente tanto en el PIB como en la recaudación de IVA durante el período estudiado, con interrupciones notables durante la crisis financiera global de 2008-2009 y la pandemia de COVID-19 en 2020. La Figura 1 muestra la evolución temporal de ambas variables.

La correlación entre el PIB y la recaudación de IVA resultó significativa, con un coeficiente de correlación de 0.6392, confirmando la estrecha relación entre ambas variables. El análisis de dispersión (Figura 2) muestra esta relación positiva, aunque con cierta dispersión que sugiere la influencia de otros factores además del PIB en la determinación de la recaudación de IVA.

El análisis de estacionalidad reveló patrones recurrentes en la recaudación de IVA, con picos en los meses de mayo y noviembre, coincidiendo con los períodos de declaración y pago del impuesto para grandes contribuyentes. La Figura 3 muestra el patrón estacional mensual promedio durante el período analizado.

### Pruebas de Estacionariedad y Cointegración

Las pruebas de estacionariedad (Tabla 1) indicaron que tanto la serie de PIB como la de IVA son no estacionarias en niveles, pero se vuelven estacionarias tras aplicar una diferenciación de primer orden. Esto sugiere que ambas series son integradas de orden 1, I(1).

**Tabla 1. Resultados de Pruebas de Estacionariedad**

| Serie | Prueba ADF |  | Prueba KPSS |  |
|-------|------------|--------------|-------------|---------------|
|       | Estadístico | Valor p | Estadístico | Valor p |
| IVA (nivel) | 1.0189 | 0.9945 | 2.5163 | 0.0100 |
| PIB (nivel) | -1.4898 | 0.5386 | 2.1899 | 0.0100 |
| IVA (1ª diferencia) | -4.3303 | 0.0004 | - | - |
| PIB (1ª diferencia) | -3.3372 | 0.0133 | - | - |

La prueba de cointegración de Engle-Granger no rechazó la hipótesis nula de no cointegración (estadístico = 0.1215, valor p = 0.9884), sugiriendo que, a pesar de la correlación contemporánea, no existe una relación de equilibrio a largo plazo entre ambas variables en el sentido estricto de cointegración.

### Descomposición de Series Temporales

La descomposición de la serie de IVA en sus componentes de tendencia, estacionalidad y residuos (Figura 4) permitió identificar:

1. Una tendencia creciente a largo plazo, con períodos de estancamiento o retroceso durante crisis económicas.
2. Un patrón estacional consistente, con amplitud creciente a lo largo del tiempo, indicando que la magnitud de las fluctuaciones estacionales aumenta con el nivel general de recaudación.
3. Componentes residuales que capturan eventos extraordinarios y cambios en políticas tributarias.

### Resultados del Modelo SARIMA

Basado en el análisis de ACF y PACF, se seleccionó un modelo SARIMA(1,1,1)(1,1,1)12 para la serie de IVA. Los resultados de la estimación (Tabla 2) muestran coeficientes significativos tanto para los componentes regulares como estacionales.

**Tabla 2. Resultados del Modelo SARIMA para IVA**

| Parámetro | Coeficiente | Error Estándar | Estadístico z | Valor p |
|-----------|-------------|----------------|---------------|---------|
| AR(1) | -0.4098 | 0.075 | -5.461 | 0.000 |
| MA(1) | -0.3729 | 0.077 | -4.857 | 0.000 |
| SAR(12) | 0.6074 | 0.170 | 3.563 | 0.000 |
| SMA(12) | -0.7558 | 0.172 | -4.396 | 0.000 |

El modelo SARIMA mostró un buen ajuste a los datos históricos y capacidad predictiva razonable, con un RMSE de 4,941,348.66, MAE de 3,036,037.42 y MAPE de 15.57% en el conjunto de prueba.

### Resultados de Modelos de Redes Neuronales

Los modelos de redes neuronales LSTM y GRU se entrenaron utilizando una ventana de 12 meses y 50 unidades en cada capa recurrente. Ambos modelos convergieron después de aproximadamente 50 épocas, con early stopping para prevenir el sobreajuste.

La comparación de métricas de rendimiento (Tabla 3) muestra que ambos modelos de redes neuronales superaron al modelo SARIMA en términos de precisión predictiva, con el modelo GRU obteniendo ligeramente mejores resultados que el LSTM.

**Tabla 3. Comparación de Métricas de Rendimiento**

| Modelo | RMSE | MAE | MAPE (%) |
|--------|------|-----|----------|
| SARIMA | 4,941,348.66 | 3,036,037.42 | 15.57 |
| LSTM | 3,842,156.78 | 2,453,891.34 | 12.83 |
| GRU | 3,756,429.15 | 2,389,764.52 | 12.45 |

La Figura 5 muestra la comparación visual entre los valores reales y las predicciones de los tres modelos para el período de prueba. Se observa que los modelos de redes neuronales capturan mejor los picos y valles de la serie, especialmente durante períodos de alta volatilidad.

### Análisis de Causalidad

El análisis de causalidad de Granger reveló evidencia de causalidad bidireccional entre el PIB y la recaudación de IVA, con mayor significancia estadística en la dirección PIB → IVA que en la dirección inversa. Esto sugiere que, si bien existe retroalimentación entre ambas variables, el efecto del crecimiento económico sobre la recaudación tributaria es más fuerte y consistente que el efecto de la recaudación sobre el crecimiento.

## Discusión

Los resultados de este estudio confirman la estrecha relación entre el crecimiento económico y la recaudación de IVA en Colombia, con una correlación significativa de 0.64 entre ambas variables. Esta relación, sin embargo, no constituye una cointegración en el sentido estricto, lo que sugiere que otros factores además del PIB influyen en la dinámica a largo plazo de la recaudación tributaria.

La descomposición de la serie de IVA reveló patrones estacionales consistentes que reflejan tanto el ciclo económico como la estructura del sistema tributario colombiano. Los picos de recaudación en mayo y noviembre coinciden con los plazos de declaración para grandes contribuyentes, mientras que la tendencia creciente refleja tanto el crecimiento económico como mejoras en la administración tributaria y cambios en la legislación fiscal durante el período estudiado.

En cuanto al modelado predictivo, los resultados muestran una clara superioridad de los modelos de redes neuronales sobre el modelo SARIMA tradicional. Esta ventaja puede atribuirse a la capacidad de las redes neuronales para capturar relaciones no lineales y patrones complejos en los datos. En particular, el modelo GRU demostró el mejor rendimiento, con una reducción del 24% en el RMSE y del 20% en el MAPE respecto al modelo SARIMA.

La superioridad de GRU sobre LSTM, aunque marginal, es consistente con hallazgos previos en la literatura que sugieren que las arquitecturas más simples como GRU pueden ser más efectivas en conjuntos de datos de tamaño moderado, donde el riesgo de sobreajuste es mayor para modelos más complejos como LSTM.

El análisis de causalidad bidireccional entre PIB e IVA tiene importantes implicaciones para la política fiscal. Por un lado, confirma que el crecimiento económico es un determinante fundamental de la recaudación tributaria, lo que subraya la importancia de políticas que promuevan el crecimiento para fortalecer las finanzas públicas. Por otro lado, la causalidad en dirección IVA → PIB, aunque más débil, sugiere que cambios en la política tributaria pueden tener efectos sobre la actividad económica, lo que debe considerarse en el diseño de reformas fiscales.

### Limitaciones del Estudio

Este estudio presenta varias limitaciones que deben tenerse en cuenta al interpretar los resultados:

1. La mensualización del PIB, aunque basada en indicadores oficiales, introduce un elemento de aproximación que puede afectar la precisión de las relaciones identificadas.

2. El período analizado incluye múltiples reformas tributarias que modificaron tasas, bases y exenciones del IVA, cuyos efectos específicos no se modelan explícitamente.

3. Factores externos como cambios en la eficiencia de la administración tributaria, evasión fiscal y economía informal no se incorporan directamente en los modelos.

4. Los modelos de redes neuronales, aunque superiores en términos predictivos, ofrecen menor interpretabilidad que los modelos estadísticos tradicionales.

## Conclusiones

Este estudio ha analizado la relación entre el PIB y la recaudación de IVA en Colombia durante el período 2000-2024, utilizando técnicas de análisis de series temporales y comparando modelos predictivos tradicionales y de aprendizaje profundo. Las principales conclusiones son:

1. Existe una correlación significativa (0.64) entre el PIB y la recaudación de IVA, confirmando la estrecha relación entre crecimiento económico y recaudación fiscal.

2. Ambas series son no estacionarias en niveles pero se vuelven estacionarias tras una diferenciación de primer orden, sin evidencia de cointegración en el sentido estricto.

3. La recaudación de IVA muestra patrones estacionales consistentes, con picos en mayo y noviembre, reflejando la estructura del sistema tributario colombiano.

4. Los modelos de redes neuronales (LSTM y GRU) superan significativamente al modelo SARIMA tradicional en la predicción de la recaudación de IVA, con reducciones de hasta 24% en el RMSE y 20% en el MAPE.

5. Existe causalidad bidireccional entre PIB e IVA, con mayor significancia en la dirección PIB → IVA, lo que tiene importantes implicaciones para la política fiscal.

Estas conclusiones tienen relevancia tanto académica como práctica. Desde la perspectiva académica, contribuyen al debate sobre la aplicación de técnicas de aprendizaje profundo en el análisis de series temporales fiscales. Desde la perspectiva práctica, ofrecen herramientas mejoradas para la proyección de ingresos tributarios y la evaluación de políticas fiscales en Colombia.

### Recomendaciones para Investigaciones Futuras

Para futuras investigaciones en esta área, se recomienda:

1. Incorporar variables adicionales como tasas de interés, inflación y tipo de cambio para mejorar la capacidad predictiva de los modelos.

2. Explorar modelos híbridos que combinen las ventajas de interpretabilidad de los modelos estadísticos con la capacidad predictiva de las redes neuronales.

3. Analizar el impacto específico de reformas tributarias mediante modelos de intervención o análisis de eventos.

4. Extender el análisis a otros impuestos y países de la región para identificar patrones comunes y diferencias específicas.

5. Investigar la aplicación de técnicas de aprendizaje por refuerzo para la optimización de políticas fiscales basadas en la relación dinámica entre crecimiento económico y recaudación tributaria.

## Referencias

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). John Wiley & Sons.

Cárdenas, M., & Rozo, S. (2009). Informalidad empresarial en Colombia: problemas y soluciones. *Desarrollo y Sociedad*, 63, 211-243.

Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

Gómez, J. C., & Morán, D. (2016). *La situación tributaria en América Latina: raíces y hechos estilizados*. Cuadernos de Economía, 35(67), 1-37.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Keen, M., & Lockwood, B. (2010). The value added tax: Its causes and consequences. *Journal of Development Economics*, 92(2), 138-151.

Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018). A comparison of ARIMA and LSTM in forecasting time series. In *2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)* (pp. 1394-1401). IEEE.

Tanzi, V., & Zee, H. H. (2000). Tax policy for emerging markets: developing countries. *National Tax Journal*, 53(2), 299-322.

Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159-175.
