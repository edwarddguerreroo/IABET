"""
Configuración para Modelos de Deep Learning AST
==============================================

Contiene todas las configuraciones y hiperparámetros para los diferentes
modelos de Deep Learning especializados en predicción de asistencias.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class BaseConfig:
    """Configuración base para todos los modelos DL."""
    
    # Configuración general
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    dtype: torch.dtype = torch.float32
    
    # Configuración de entrenamiento
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    
    # Configuración de validación
    validation_split: float = 0.2
    test_split: float = 0.1
    cv_folds: int = 5
    
    # Configuración de datos
    sequence_length: int = 10  # Últimos N juegos
    input_features: int = 259  # Número real de features numéricas generadas
    target_feature: str = "AST"
    
    # Configuración de regularización
    dropout_rate: float = 0.2
    batch_norm: bool = True
    gradient_clip: float = 1.0


@dataclass
class TransformerConfig(BaseConfig):
    """Configuración específica para el modelo Transformer."""
    
    # Arquitectura Transformer
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    
    # Configuración de embedding
    embedding_dim: int = 128
    max_sequence_length: int = 82  # Temporada completa
    
    # Configuración de cabezas
    prediction_head_dims: List[int] = None
    
    def __post_init__(self):
        if self.prediction_head_dims is None:
            self.prediction_head_dims = [64, 32, 1]


@dataclass
class GNNConfig(BaseConfig):
    """Configuración específica para Graph Neural Networks."""
    
    # Arquitectura GNN - CORREGIDA para compatibilidad con features reales
    node_features: int = 259  # Usar el mismo número que input_features
    edge_features: int = 0    # Sin features de aristas por simplicidad
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    gnn_type: str = "GCN"  # GCN, GAT, GraphSAGE
    
    # Configuración de atención (para GAT)
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Configuración de agregación
    aggregation_type: str = "mean"  # mean, max, sum, attention
    
    # Configuración de grafo
    max_nodes: int = 50  # Máximo jugadores + equipos en el grafo
    self_loops: bool = True
    
    def __post_init__(self):
        # Asegurar que node_features coincida con input_features
        self.node_features = self.input_features


@dataclass
class LSTMConfig(BaseConfig):
    """Configuración específica para LSTM con Attention."""
    
    # Arquitectura LSTM
    hidden_size: int = 128
    num_layers: int = 3
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    
    # Configuración de atención
    attention_dim: int = 256  # hidden_size * 2 si bidirectional
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Configuración de predictor
    predictor_dims: List[int] = None
    
    def __post_init__(self):
        if self.predictor_dims is None:
            self.predictor_dims = [256, 128, 64, 1]
        
        # Ajustar attention_dim si es bidirectional
        if self.bidirectional:
            self.attention_dim = self.hidden_size * 2


@dataclass
class VAEConfig(BaseConfig):
    """Configuración específica para Variational Autoencoder."""
    
    # Arquitectura VAE
    latent_dim: int = 20
    encoder_dims: List[int] = None
    decoder_dims: List[int] = None
    
    # Configuración de pérdida
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    prediction_weight: float = 10.0
    
    # Configuración de predictor
    predictor_dims: List[int] = None
    
    def __post_init__(self):
        if self.encoder_dims is None:
            self.encoder_dims = [128, 64]
        if self.decoder_dims is None:
            self.decoder_dims = [64, 128]
        if self.predictor_dims is None:
            self.predictor_dims = [32, 16, 1]


@dataclass
class EnsembleConfig(BaseConfig):
    """Configuración específica para Ensemble Especializado."""
    
    # Configuración de expertos
    num_experts: int = 4
    expert_types: List[str] = None
    
    # Configuración de meta-learner
    meta_learner_dims: List[int] = None
    meta_learning_rate: float = 1e-4
    
    # Configuración de pesos
    expert_weights: Optional[List[float]] = None
    adaptive_weights: bool = True
    
    def __post_init__(self):
        if self.expert_types is None:
            self.expert_types = ["temporal", "team", "individual", "matchup"]
        if self.meta_learner_dims is None:
            self.meta_learner_dims = [64, 32, 1]


@dataclass
class HybridConfig(BaseConfig):
    """Configuración específica para el modelo Híbrido."""
    
    # Configuraciones de componentes
    transformer_config: TransformerConfig = None
    gnn_config: GNNConfig = None
    
    # Configuración de fusión
    fusion_dims: List[int] = None
    fusion_type: str = "concat"  # concat, attention, gated
    
    # Configuración de pesos de componentes
    temporal_weight: float = 0.6
    graph_weight: float = 0.4
    
    def __post_init__(self):
        if self.transformer_config is None:
            self.transformer_config = TransformerConfig(
                d_model=64, nhead=4, num_encoder_layers=3
            )
        if self.gnn_config is None:
            self.gnn_config = GNNConfig(
                hidden_dim=32, num_gnn_layers=2
            )
        if self.fusion_dims is None:
            self.fusion_dims = [96, 64, 32, 1]  # 64 + 32 = 96


class DLConfig:
    """Clase principal de configuración que contiene todas las configuraciones."""
    
    def __init__(self):
        self.base = BaseConfig()
        self.transformer = TransformerConfig()
        self.gnn = GNNConfig()
        self.lstm = LSTMConfig()
        self.vae = VAEConfig()
        self.ensemble = EnsembleConfig()
        self.hybrid = HybridConfig()
    
    def get_config(self, model_type: str) -> BaseConfig:
        """Obtiene la configuración para un tipo de modelo específico."""
        config_map = {
            "transformer": self.transformer,
            "gnn": self.gnn,
            "lstm": self.lstm,
            "vae": self.vae,
            "ensemble": self.ensemble,
            "hybrid": self.hybrid
        }
        
        if model_type not in config_map:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Asegurar que todas las configuraciones tengan el input_features correcto
        config = config_map[model_type]
        config.input_features = self.base.input_features
        
        return config
    
    def update_config(self, model_type: str, **kwargs):
        """Actualiza la configuración de un modelo específico."""
        config = self.get_config(model_type)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Parámetro no válido: {key}")
    
    def to_dict(self) -> Dict:
        """Convierte todas las configuraciones a diccionario."""
        return {
            "base": self.base.__dict__,
            "transformer": self.transformer.__dict__,
            "gnn": self.gnn.__dict__,
            "lstm": self.lstm.__dict__,
            "vae": self.vae.__dict__,
            "ensemble": self.ensemble.__dict__,
            "hybrid": self.hybrid.__dict__
        }


# Configuraciones predefinidas para diferentes escenarios
_quick_config = DLConfig()
_quick_config.base.input_features = 148  # Actualizar número de features
_quick_config.update_config("transformer", num_encoder_layers=2, d_model=64)
_quick_config.update_config("gnn", num_gnn_layers=2, hidden_dim=32)
_quick_config.update_config("lstm", num_layers=2, hidden_size=64)

_performance_config = DLConfig()
_performance_config.base.input_features = 148  # Actualizar número de features
_performance_config.update_config("transformer", num_encoder_layers=8, d_model=256)
_performance_config.update_config("gnn", num_gnn_layers=4, hidden_dim=128)
_performance_config.update_config("lstm", num_layers=4, hidden_size=256)

_production_config = DLConfig()
_production_config.base.input_features = 148  # Actualizar número de features
_production_config.update_config("transformer", num_encoder_layers=6, d_model=128)
_production_config.update_config("gnn", num_gnn_layers=3, hidden_dim=64)
_production_config.update_config("lstm", num_layers=3, hidden_size=128)

# Configuraciones específicas por modelo para acceso directo
QUICK_CONFIG = {
    "transformer": _quick_config.get_config("transformer"),
    "gnn": _quick_config.get_config("gnn"),
    "lstm": _quick_config.get_config("lstm"),
    "vae": _quick_config.get_config("vae"),
    "ensemble": _quick_config.get_config("ensemble"),
    "hybrid": _quick_config.get_config("hybrid")
}

PERFORMANCE_CONFIG = {
    "transformer": _performance_config.get_config("transformer"),
    "gnn": _performance_config.get_config("gnn"),
    "lstm": _performance_config.get_config("lstm"),
    "vae": _performance_config.get_config("vae"),
    "ensemble": _performance_config.get_config("ensemble"),
    "hybrid": _performance_config.get_config("hybrid")
}

PRODUCTION_CONFIG = {
    "transformer": _production_config.get_config("transformer"),
    "gnn": _production_config.get_config("gnn"),
    "lstm": _production_config.get_config("lstm"),
    "vae": _production_config.get_config("vae"),
    "ensemble": _production_config.get_config("ensemble"),
    "hybrid": _production_config.get_config("hybrid")
} 