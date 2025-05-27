"""
Ejecutor del Dashboard PIB-IVA Colombia
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🇨🇴 EJECUTANDO DASHBOARD PIB-IVA COLOMBIA")
    print("=" * 50)
    
    # Instalar dependencias si es necesario
    try:
        import streamlit
        print("✅ Streamlit disponible")
    except ImportError:
        print("📦 Instalando Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas", "numpy", "scikit-learn"])
    
    # Ejecutar dashboard
    print("🚀 Iniciando dashboard en http://localhost:8501")
    print("⏹️ Para detener: Ctrl+C")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido")

if __name__ == "__main__":
    main()
