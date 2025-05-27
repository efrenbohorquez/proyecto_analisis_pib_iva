"""
🚀 EJECUTOR INMEDIATO DEL DASHBOARD PIB-IVA COLOMBIA
Ejecuta el dashboard sin verificaciones previas
"""

import subprocess
import sys
import os
from pathlib import Path

def ejecutar_inmediato():
    """Ejecutar dashboard inmediatamente"""
    
    print("🇨🇴 DASHBOARD PIB-IVA COLOMBIA 2000-2024")
    print("=" * 50)
    print("🚀 EJECUTANDO INMEDIATAMENTE...")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Verificar que streamlit_app.py existe
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py no encontrado")
        print("📁 Archivos en directorio:")
        for file in Path(".").glob("*.py"):
            print(f"   - {file.name}")
        return False
    
    # Instalar streamlit si no está disponible
    try:
        import streamlit
        print("✅ Streamlit disponible")
    except ImportError:
        print("📦 Instalando Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Ejecutar dashboard
    print("\n🌐 Iniciando en: http://localhost:8501")
    print("📱 Se abrirá automáticamente en el navegador")
    print("⏹️ Para detener: Ctrl+C")
    print("\n" + "=" * 50)
    
    try:
        # Comando streamlit con configuración optimizada
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido")
        print("✅ Ejecución finalizada")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Alternativa manual:")
        print("streamlit run streamlit_app.py")
        return False
    
    return True

if __name__ == "__main__":
    ejecutar_inmediato()
