"""
Configuración centralizada del proyecto de análisis PIB-IVA
Sigue patrones de configuración modernos para ciencia de datos
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    host: str = "localhost"
    port: int = 5432
    name: str = "pib_iva_db"
    user: str = "analytics_user"
    password: str = os.getenv("DB_PASSWORD", "")

@dataclass
class ModelConfig:
    """Configuración de modelos de ML"""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Hiperparámetros por modelo
    random_forest: Dict = None
    gradient_boosting: Dict = None
    neural_network: Dict = None
    
    def __post_init__(self):
        if self.random_forest is None:
            self.random_forest = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
        
        if self.gradient_boosting is None:
            self.gradient_boosting = {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': self.random_state
            }
        
        if self.neural_network is None:
            self.neural_network = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'random_state': self.random_state,
                'early_stopping': True
            }

@dataclass
class VisualizationConfig:
    """Configuración de visualizaciones académicas"""
    # Tema académico
    theme: str = "plotly_white"
    color_palette: List[str] = None
    figure_width: int = 1200
    figure_height: int = 800
    dpi: int = 300
    
    # Fuentes académicas
    font_family: str = "Arial"
    font_size: int = 14
    title_font_size: int = 18
    
    # Colores específicos
    color_primary: str = "#2E86AB"
    color_secondary: str = "#A23B72"
    color_accent: str = "#F18F01"
    color_success: str = "#C73E1D"
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                self.color_primary,
                self.color_secondary, 
                self.color_accent,
                self.color_success,
                "#85C1E9",
                "#F8C471",
                "#82E0AA",
                "#F1948A"
            ]

@dataclass
class ProjectConfig:
    """Configuración principal del proyecto"""
    # Rutas del proyecto
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = None
    models_dir: Path = None
    output_dir: Path = None
    logs_dir: Path = None
    docs_dir: Path = None
    
    # Configuraciones específicas
    database: DatabaseConfig = None
    models: ModelConfig = None
    visualization: VisualizationConfig = None
    
    # Configuración de logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        # Inicializar rutas
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.docs_dir is None:
            self.docs_dir = self.project_root / "docs"
        
        # Crear directorios si no existen
        for directory in [self.data_dir, self.models_dir, self.output_dir, self.logs_dir, self.docs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Inicializar sub-configuraciones
        if self.database is None:
            self.database = DatabaseConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()

# Instancia global de configuración
config = ProjectConfig()
