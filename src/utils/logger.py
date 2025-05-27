"""
Sistema de logging moderno para el proyecto PIB-IVA
Implementa patrones de logging para ciencia de datos
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class DataScienceLogger:
    """Logger especializado para proyectos de ciencia de datos"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None, level: str = "INFO"):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Crear logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Evitar duplicar handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configurar handlers para consola y archivo"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para archivo
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_data_quality(self, dataset_name: str, shape: tuple, missing_values: int, duplicates: int):
        """Log específico para calidad de datos"""
        self.logger.info(f"CALIDAD DE DATOS - {dataset_name}")
        self.logger.info(f"  Forma del dataset: {shape}")
        self.logger.info(f"  Valores faltantes: {missing_values}")
        self.logger.info(f"  Duplicados: {duplicates}")
    
    def log_model_performance(self, model_name: str, metrics: dict):
        """Log específico para performance de modelos"""
        self.logger.info(f"PERFORMANCE MODELO - {model_name}")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
    
    def log_visualization_created(self, viz_name: str, file_path: Optional[str] = None):
        """Log específico para visualizaciones creadas"""
        msg = f"VISUALIZACIÓN CREADA - {viz_name}"
        if file_path:
            msg += f" -> {file_path}"
        self.logger.info(msg)
    
    def __getattr__(self, name):
        """Delegar métodos al logger interno"""
        return getattr(self.logger, name)

def get_logger(name: str) -> DataScienceLogger:
    """Factory function para crear loggers"""
    return DataScienceLogger(name)
