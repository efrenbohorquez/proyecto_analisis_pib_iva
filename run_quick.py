"""
Ejecutor rápido del dashboard PIB-IVA Colombia
"""

import subprocess
import sys
import os

def main():
    print("🇨🇴 EJECUTANDO DASHBOARD PIB-IVA COLOMBIA")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    target_dir = r"d:\Downloads\proyecto_analisis_pib_iva"
    
    try:
        os.chdir(target_dir)
        print(f"✅ Directorio: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Error cambiando directorio: {e}")
        return
    
    # Verificar que streamlit_app.py existe
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py no encontrado")
        return
    
    # Instalar streamlit si es necesario
    try:
        import streamlit
        print("✅ Streamlit disponible")
    except ImportError:
        print("📦 Instalando Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Ejecutar dashboard
    print("🚀 Iniciando dashboard...")
    print("🌐 URL: http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
