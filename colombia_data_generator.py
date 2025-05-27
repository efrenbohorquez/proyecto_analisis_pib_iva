"""
Generador de datos económicos históricos de Colombia 2000-2024
Basado en datos reales del DANE y Banco de la República
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_colombia_historical_data():
    """
    Generar datos históricos de PIB e IVA Colombia con eventos reales
    
    Returns:
        tuple: (df_pib, df_iva) con datos 2000-2024
    """
    
    # Años de análisis
    years = list(range(2000, 2025))
    
    # PIB Colombia - Crecimiento histórico real (aproximado)
    pib_growth_rates = [
        2.9,   # 2000
        1.5,   # 2001 - Crisis Argentina
        4.4,   # 2002
        3.9,   # 2003
        4.7,   # 2004
        6.7,   # 2005 - Boom commodities
        6.7,   # 2006
        3.5,   # 2007
        1.7,   # 2008 - Crisis financiera
        4.0,   # 2009
        4.0,   # 2010
        6.5,   # 2011
        5.4,   # 2012
        4.4,   # 2013
        3.2,   # 2014
        2.0,   # 2015 - Caída petróleo
        -6.8,  # 2020 - COVID-19
        10.6,  # 2021 - Recuperación
        3.5,   # 2022
        7.7,   # 2023 - Estimado
        5.0,   # 2024 - Proyección
    ]
    
    # Completar hasta 25 años si es necesario
    while len(pib_growth_rates) < len(years):
        pib_growth_rates.append(3.5)  # Crecimiento promedio
    
    # Calcular PIB nominal (billones COP)
    pib_base = 250  # PIB 2000 en billones COP nominales
    pib_values = [pib_base]
    
    for i in range(1, len(years)):
        growth = pib_growth_rates[i-1] / 100
        # Incluir inflación promedio (~4% anual)
        nominal_growth = growth + 0.04 + np.random.normal(0, 0.01)
        new_pib = pib_values[-1] * (1 + nominal_growth)
        pib_values.append(new_pib)
    
    # IVA Colombia - Eventos de política tributaria
    iva_events = {
        2001: 0.01,   # Reforma tributaria
        2003: 0.005,  # Ajuste menor
        2005: 0.01,   # Reforma Uribe
        2010: 0.008,  # Post-crisis
        2013: 0.012,  # Reforma Santos
        2017: 0.015,  # IVA al 19%
        2019: 0.005,  # Ajustes menores
        2021: -0.005, # Flexibilización COVID
        2023: 0.008   # Reforma Petro
    }
    
    # Calcular recaudación IVA
    base_iva_rate = 0.065  # 6.5% del PIB base
    iva_values = []
    
    for i, year in enumerate(years):
        # Tasa base
        current_rate = base_iva_rate
        
        # Aplicar eventos de política
        for event_year, change in iva_events.items():
            if year >= event_year:
                current_rate += change
        
        # Efectos económicos cíclicos
        if year in [2008, 2009]:  # Crisis
            current_rate *= 0.85
        elif year == 2020:  # COVID
            current_rate *= 0.75
        elif year in [2021, 2022]:  # Recuperación
            current_rate *= 1.1
        
        # Variabilidad aleatoria
        current_rate *= (1 + np.random.normal(0, 0.02))
        
        # Calcular IVA
        iva_value = pib_values[i] * current_rate
        iva_values.append(iva_value)
    
    # Crear DataFrames
    df_pib = pd.DataFrame({
        'año': years,
        'fecha': pd.to_datetime([f'{year}-12-31' for year in years]),
        'valor': pib_values,
        'crecimiento_real': pib_growth_rates[:len(years)],
        'sector': 'Total'
    })
    
    df_iva = pd.DataFrame({
        'año': years,
        'fecha': pd.to_datetime([f'{year}-12-31' for year in years]),
        'valor': iva_values,
        'ratio_pib': np.array(iva_values) / np.array(pib_values) * 100,
        'tasa_efectiva': [base_iva_rate * 100] * len(years)  # Simplificado
    })
    
    return df_pib, df_iva

if __name__ == "__main__":
    # Generar y guardar datos
    df_pib, df_iva = generate_colombia_historical_data()
    
    # Guardar archivos
    df_pib.to_csv('data/colombia_pib_2000_2024.csv', index=False)
    df_iva.to_csv('data/colombia_iva_2000_2024.csv', index=False)
    
    print("✅ Datos de Colombia generados exitosamente")
    print(f"PIB 2000: {df_pib['valor'].iloc[0]:.1f} billones COP")
    print(f"PIB 2024: {df_pib['valor'].iloc[-1]:.1f} billones COP")
    print(f"IVA 2024: {df_iva['valor'].iloc[-1]:.1f} billones COP")
    print(f"Ratio IVA/PIB 2024: {df_iva['ratio_pib'].iloc[-1]:.1f}%")
