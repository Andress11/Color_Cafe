# Importar las clases principales
from .preprocessing import ImagePreprocessor
from .read_features import FeatureExtractor
from .segmentation import ImageSegmenter

# Definir metadatos del paquete
__version__ = '1.0.0'
__author__ = 'Jorge A. Ramírez, Jose D. Ardila, Andrés F. Cerón'
__email__ = 'jorge.ramirez@profesores.uamerica.edu.co'
__description__ = 'Calsificación de Café'

# Definir qué se exporta cuando se importa el paquete
__all__ = [
    "ImagePreprocessor",
    "FeatureExtractor",
    "ImageSegmenter",
]
