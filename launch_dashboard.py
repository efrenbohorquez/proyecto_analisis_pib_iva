"""
Lanzador inteligente del Dashboard PIB-IVA Colombia
Verifica dependencias e instala automáticamente si faltan
"""

import subprocess
import sys
import os
import importlib

def check_and_install_packages():
    """Verificar e instalar paquetes necesarios"""
    
    packages = {
        'streamlit': 'streamlit',
        'plotly': 'plotly', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels',
        'pmdarima': 'pmdarima',
        'scipy': 'scipy'
    }
    
    print("🔍 Verificando dependencias...")
    
    missing_packages = []
    installed_packages = []
    
    for module, package in packages.items():
        try:
            importlib.import_module(module)
            installed_packages.append(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
    
    # Mostrar estado
    for pkg in installed_packages:
        print(f"   {pkg}")
    
    if missing_packages:
        print(f"\n📦 Instalando paquetes faltantes: {', '.join(missing_packages)}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + missing_packages)
            print("✅ Instalación completada")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando paquetes: {e}")
            return False
    else:
        print("✅ Todas las dependencias están instaladas")
    
    return True

def launch_streamlit():
    """Lanzar aplicación Streamlit"""
    
    print("\n🚀 LANZANDO DASHBOARD PIB-IVA COLOMBIA")
    print("=" * 50)
    print("📊 Análisis PIB-IVA Colombia 2000-2024")
    print("🔮 Predicción IVA 2025")
    print("📈 Modelos: ARIMA, SARIMA, SARIMAX, VAR, ML")
    print("=" * 50)
    print("🌐 URL: http://localhost:8501")
    print("⏹️ Para detener: Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {e}")

def main():
    """Función principal"""
    
    print("🇨🇴 DASHBOARD PIB-IVA COLOMBIA 2000-2024")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Directorio: {os.getcwd()}")
    
    # Verificar que streamlit_app.py existe
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py no encontrado en el directorio actual")
        print("📁 Archivos disponibles:")
        for file in os.listdir("."):
            if file.endswith(".py"):
                print(f"   - {file}")
        return
    
    # Verificar e instalar dependencias
    if check_and_install_packages():
        # Lanzar dashboard
        launch_streamlit()
    else:
        print("❌ No se pudieron instalar las dependencias necesarias")

if __name__ == "__main__":
    main()
