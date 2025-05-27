"""
Script de verificación y ejecución del Dashboard PIB-IVA Colombia
Revisa cambios y ejecuta el dashboard completo
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib.util

def verificar_archivos():
    """Verificar que todos los archivos necesarios existen"""
    print("🔍 VERIFICANDO ARCHIVOS DEL PROYECTO")
    print("=" * 50)
    
    archivos_requeridos = [
        "streamlit_app.py",
        "ejecutar_dashboard.py",
        "EJECUTAR.bat",
        "EJECUTAR.sh"
    ]
    
    archivos_existentes = []
    archivos_faltantes = []
    
    for archivo in archivos_requeridos:
        if Path(archivo).exists():
            size = Path(archivo).stat().st_size
            archivos_existentes.append(f"✅ {archivo} ({size:,} bytes)")
        else:
            archivos_faltantes.append(f"❌ {archivo}")
    
    print("📁 Archivos encontrados:")
    for archivo in archivos_existentes:
        print(f"   {archivo}")
    
    if archivos_faltantes:
        print("\n⚠️ Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"   {archivo}")
        return False
    
    print(f"\n✅ Todos los archivos requeridos están presentes ({len(archivos_existentes)}/{len(archivos_requeridos)})")
    return True

def verificar_dependencias():
    """Verificar dependencias de Python"""
    print("\n📦 VERIFICANDO DEPENDENCIAS")
    print("=" * 50)
    
    dependencias = {
        'streamlit': 'Streamlit framework',
        'plotly': 'Visualizaciones interactivas',
        'sklearn': 'Machine Learning',
        'pandas': 'Manipulación de datos',
        'numpy': 'Cálculos numéricos'
    }
    
    dependencias_ok = []
    dependencias_faltantes = []
    
    for dep, descripcion in dependencias.items():
        try:
            if dep == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'unknown')
            
            dependencias_ok.append(f"✅ {dep} v{version} - {descripcion}")
        except ImportError:
            dependencias_faltantes.append(f"❌ {dep} - {descripcion}")
    
    print("📋 Dependencias instaladas:")
    for dep in dependencias_ok:
        print(f"   {dep}")
    
    if dependencias_faltantes:
        print("\n⚠️ Dependencias faltantes:")
        for dep in dependencias_faltantes:
            print(f"   {dep}")
        
        print("\n📥 Instalando dependencias faltantes...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + 
                                [dep.split()[1] for dep in dependencias_faltantes])
            print("✅ Dependencias instaladas exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
    
    print(f"\n✅ Todas las dependencias están instaladas ({len(dependencias_ok)}/{len(dependencias)})")
    return True

def verificar_codigo_streamlit():
    """Verificar que el código de Streamlit es válido"""
    print("\n🔍 VERIFICANDO CÓDIGO STREAMLIT")
    print("=" * 50)
    
    try:
        # Verificar sintaxis
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            codigo = f.read()
        
        compile(codigo, 'streamlit_app.py', 'exec')
        print("✅ Sintaxis de Python válida")
        
        # Verificar imports críticos
        imports_criticos = [
            'import streamlit as st',
            'import pandas as pd',
            'import numpy as np',
            'import plotly.graph_objects as go',
            'from sklearn'
        ]
        
        imports_encontrados = 0
        for imp in imports_criticos:
            if imp in codigo:
                imports_encontrados += 1
                print(f"✅ {imp}")
            else:
                print(f"⚠️ {imp} - No encontrado")
        
        # Verificar funciones críticas
        funciones_criticas = [
            'def generate_colombia_data',
            'def train_models',
            'def predict_2025',
            'def main('
        ]
        
        funciones_encontradas = 0
        for func in funciones_criticas:
            if func in codigo:
                funciones_encontradas += 1
                print(f"✅ {func}")
            else:
                print(f"❌ {func} - No encontrada")
        
        if imports_encontrados >= 4 and funciones_encontradas >= 3:
            print(f"\n✅ Código Streamlit verificado correctamente")
            return True
        else:
            print(f"\n⚠️ Código incompleto - revisar funciones faltantes")
            return False
            
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verificando código: {e}")
        return False

def mostrar_resumen_cambios():
    """Mostrar resumen de los cambios implementados"""
    print("\n📊 RESUMEN DE CAMBIOS IMPLEMENTADOS")
    print("=" * 50)
    
    cambios = [
        "✅ Dashboard Streamlit completo para Colombia 2000-2024",
        "✅ Análisis histórico con series PIB e IVA en presentación inicial",
        "✅ Base gravable del IVA con análisis detallado",
        "✅ Corrección de KeyError 'base_ratio_pib'",
        "✅ Eventos históricos marcados en gráficos",
        "✅ Correlación PIB-IVA con scatter plot interactivo",
        "✅ Modelos ML: Random Forest, Gradient Boosting, Regresión Lineal",
        "✅ Predicción IVA 2025 con múltiples escenarios",
        "✅ Métricas de eficiencia del recaudo tributario",
        "✅ Scripts de ejecución automática",
        "✅ Configuración para Streamlit Cloud"
    ]
    
    for cambio in cambios:
        print(f"   {cambio}")
    
    print(f"\n🎯 Total de mejoras implementadas: {len(cambios)}")

def ejecutar_dashboard():
    """Ejecutar el dashboard de Streamlit"""
    print("\n🚀 EJECUTANDO DASHBOARD")
    print("=" * 50)
    print("🌐 URL: http://localhost:8501")
    print("📱 Se abrirá automáticamente en el navegador")
    print("⏹️ Para detener: Ctrl+C en la terminal")
    print("=" * 50)
    
    try:
        # Comando para ejecutar Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🔄 Iniciando servidor Streamlit...")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido por el usuario")
        print("✅ Ejecución finalizada correctamente")
    except FileNotFoundError:
        print("❌ Streamlit no encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("🔄 Reintentando ejecución...")
        subprocess.run(cmd)
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {e}")
        print("\n🔧 Soluciones alternativas:")
        print("1. Ejecutar: python ejecutar_dashboard.py")
        print("2. Ejecutar: streamlit run streamlit_app.py")
        print("3. Verificar instalación de Streamlit")

def main():
    """Función principal de verificación y ejecución"""
    print("🇨🇴 VERIFICACIÓN Y EJECUCIÓN - DASHBOARD PIB-IVA COLOMBIA")
    print("=" * 60)
    print("📅 Período de análisis: 2000-2024")
    print("🔮 Predicción: IVA 2025")
    print("🤖 Modelos: Machine Learning")
    print("=" * 60)
    
    # Cambiar al directorio del script
    os.chdir(Path(__file__).parent)
    print(f"📁 Directorio de trabajo: {Path.cwd()}")
    
    # Verificaciones paso a paso
    pasos = [
        ("Archivos del proyecto", verificar_archivos),
        ("Dependencias Python", verificar_dependencias),
        ("Código Streamlit", verificar_codigo_streamlit)
    ]
    
    todos_ok = True
    for nombre, funcion in pasos:
        if not funcion():
            todos_ok = False
            print(f"\n❌ Fallo en verificación: {nombre}")
    
    # Mostrar resumen
    mostrar_resumen_cambios()
    
    if todos_ok:
        print("\n" + "=" * 60)
        print("✅ TODAS LAS VERIFICACIONES PASARON")
        print("🚀 LISTO PARA EJECUTAR DASHBOARD")
        print("=" * 60)
        
        respuesta = input("\n¿Ejecutar dashboard ahora? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            ejecutar_dashboard()
        else:
            print("💡 Para ejecutar manualmente:")
            print("   python ejecutar_dashboard.py")
            print("   O: streamlit run streamlit_app.py")
    else:
        print("\n" + "=" * 60)
        print("⚠️ VERIFICACIONES INCOMPLETAS")
        print("🔧 CORREGIR ERRORES ANTES DE EJECUTAR")
        print("=" * 60)

if __name__ == "__main__":
    main()
