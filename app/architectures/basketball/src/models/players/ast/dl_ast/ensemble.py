"""
Ensemble Especializado para Predicción de Asistencias
====================================================

Implementa un ensemble de modelos especializados que se enfocan en
diferentes aspectos del juego para maximizar la precisión en AST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .base import BaseDLModel, MLPBlock, initialize_weights
from .transformer import BasketballTransformer
from .lstm_attention import BiLSTMAttention
from .gnn import PlayerTeamGNN
from .vae import BasketballVAE

logger = logging.getLogger(__name__)


class SpecializedEnsemble(BaseDLModel):
    """
    Ensemble de expertos especializados para predicción de asistencias.
    
    Expertos:
    1. Temporal Expert: Especializado en patrones temporales (Transformer)
    2. Team Expert: Especializado en dinámicas de equipo (GNN)
    3. Individual Expert: Especializado en características individuales (MLP)
    4. Matchup Expert: Especializado en matchups específicos (LSTM)
    """
    
    def __init__(self, config):
        """
        Inicializa el ensemble especializado.
        
        Args:
            config: Configuración del modelo (EnsembleConfig)
        """
        super(SpecializedEnsemble, self).__init__(config, "SpecializedEnsemble")
        
        self.num_experts = config.num_experts
        self.expert_types = config.expert_types
        self.adaptive_weights = config.adaptive_weights
        
        # Crear expertos especializados
        self.experts = nn.ModuleDict()
        self._build_experts(config)
        
        # Meta-learner para combinar expertos
        self.meta_learner = self._build_meta_learner(config)
        
        # Pesos adaptativos si están habilitados
        if self.adaptive_weights:
            self.weight_predictor = self._build_weight_predictor(config)
        
        # Siempre inicializar expert_weights como fallback
        expert_weights = getattr(config, 'expert_weights', None) or [1.0 / self.num_experts] * self.num_experts
        self.register_buffer('expert_weights', torch.tensor(expert_weights, dtype=torch.float32))
        
        # Inicializar pesos
        self.apply(initialize_weights)
        
        # logger.info(f"Ensemble inicializado: {self.num_experts} expertos, "
        #            f"tipos={self.expert_types}, adaptive_weights={self.adaptive_weights}")  # Reducir verbosidad
    
    def _build_experts(self, config):
        """Construye los expertos especializados."""
        
        for i, expert_type in enumerate(self.expert_types):
            if expert_type == "temporal":
                # Experto temporal usando Transformer simplificado
                expert = TemporalExpert(config)
            elif expert_type == "team":
                # Experto de equipo usando GNN simplificado
                expert = TeamExpert(config)
            elif expert_type == "individual":
                # Experto individual usando MLP
                expert = IndividualExpert(config)
            elif expert_type == "matchup":
                # Experto de matchup usando LSTM
                expert = MatchupExpert(config)
            else:
                # Experto por defecto
                expert = IndividualExpert(config)
            
            self.experts[f'expert_{i}_{expert_type}'] = expert
    
    def _build_meta_learner(self, config) -> nn.Module:
        """Construye el meta-learner."""
        meta_dims = config.meta_learner_dims
        layers = []
        
        input_dim = self.num_experts  # Una predicción por experto
        for output_dim in meta_dims[:-1]:
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='relu'
            ))
            input_dim = output_dim
        
        # Capa final
        layers.append(nn.Linear(input_dim, meta_dims[-1]))
        
        return nn.Sequential(*layers)
    
    def _build_weight_predictor(self, config) -> nn.Module:
        """Construye el predictor de pesos adaptativos."""
        return nn.Sequential(
            MLPBlock(config.input_features, 64, dropout=config.dropout_rate),
            MLPBlock(64, 32, dropout=config.dropout_rate),
            nn.Linear(32, self.num_experts),
            nn.Softmax(dim=1)  # Pesos normalizados
        )
    
    def forward(self, x: torch.Tensor, 
                additional_data: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass del ensemble.
        
        Args:
            x: Tensor de entrada [batch_size, features] o [batch_size, seq_len, features]
            additional_data: Datos adicionales para expertos específicos
            
        Returns:
            Predicciones de asistencias [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Obtener predicciones de cada experto
        expert_predictions = []
        
        for expert_name, expert in self.experts.items():
            try:
                # Todos los expertos usan entrada estándar por ahora
                # El TeamExpert se simplifica para no necesitar graph_data
                pred = expert(x)
                
                # Asegurar que la predicción tenga la forma correcta [batch_size, 1]
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                elif pred.dim() > 2:
                    pred = pred.view(batch_size, -1)
                    if pred.size(1) != 1:
                        pred = pred[:, :1]  # Tomar solo la primera columna
                
                expert_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error en experto {expert_name}: {e}")
                # Predicción por defecto en caso de error
                expert_predictions.append(torch.zeros(batch_size, 1, device=x.device))
        
        # Verificar que todas las predicciones tengan la forma correcta
        for i, pred in enumerate(expert_predictions):
            if pred.size(1) != 1:
                expert_predictions[i] = pred[:, :1] if pred.size(1) > 1 else pred.unsqueeze(1)
        
        # Stack predicciones - cada experto debe devolver [batch_size, 1]
        expert_preds = torch.cat(expert_predictions, dim=1)  # [batch_size, num_experts]
        
        # Verificar dimensiones
        if expert_preds.size(1) != self.num_experts:
            logger.error(f"Dimensión incorrecta: expert_preds={expert_preds.shape}, esperado=[{batch_size}, {self.num_experts}]")
            # Usar solo predicción simple sin meta-learning
            return torch.mean(expert_preds, dim=1, keepdim=True)
        
        # Calcular pesos (simplificado para evitar errores)
        if self.adaptive_weights and x.dim() == 2:
            try:
                weights = self.weight_predictor(x)  # [batch_size, num_experts]
            except Exception as e:
                logger.warning(f"Error en weight_predictor: {e}, usando pesos uniformes")
                weights = torch.ones(batch_size, self.num_experts, device=x.device) / self.num_experts
        else:
            # Usar pesos fijos
            weights = self.expert_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Combinación ponderada
        weighted_prediction = torch.sum(expert_preds * weights, dim=1, keepdim=True)
        
        # Meta-learning simplificado
        try:
            meta_prediction = self.meta_learner(expert_preds)
        except Exception as e:
            logger.warning(f"Error en meta_learner: {e}, usando predicción ponderada")
            meta_prediction = weighted_prediction
        
        # Combinar predicción ponderada con meta-predicción
        final_prediction = 0.7 * weighted_prediction + 0.3 * meta_prediction
        
        return final_prediction
    
    def get_expert_predictions(self, x: torch.Tensor, 
                             additional_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Obtiene predicciones individuales de cada experto.
        
        Args:
            x: Tensor de entrada
            additional_data: Datos adicionales
            
        Returns:
            Diccionario con predicciones por experto
        """
        self.eval()
        predictions = {}
        
        with torch.no_grad():
            for expert_name, expert in self.experts.items():
                try:
                    if "team" in expert_name and additional_data and "graph_data" in additional_data:
                        pred = expert(x, additional_data["graph_data"])
                    else:
                        pred = expert(x)
                    
                    predictions[expert_name] = pred.cpu().numpy()
                except Exception as e:
                    logger.warning(f"Error obteniendo predicción de {expert_name}: {e}")
                    predictions[expert_name] = None
        
        return predictions
    
    def get_expert_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene los pesos de cada experto para una entrada dada.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Pesos de expertos [batch_size, num_experts]
        """
        if self.adaptive_weights:
            input_for_weights = x.view(x.size(0), -1) if x.dim() > 2 else x
            return self.weight_predictor(input_for_weights)
        else:
            return self.expert_weights.unsqueeze(0).expand(x.size(0), -1)


class TemporalExpert(nn.Module):
    """Experto especializado en patrones temporales."""
    
    def __init__(self, config):
        super(TemporalExpert, self).__init__()
        
        # Transformer simplificado para patrones temporales
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Embedding de entrada
        self.input_embedding = nn.Linear(config.input_features, 64)
        
        # Predictor (simplificado)
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del experto temporal."""
        if x.dim() == 2:
            # Si es 2D, agregar dimensión temporal
            x = x.unsqueeze(1)  # [batch_size, 1, features]
        
        # Embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, 64]
        
        # Transformer encoding
        encoded = self.temporal_encoder(x)  # [batch_size, seq_len, 64]
        
        # Usar último timestep
        final_state = encoded[:, -1, :]  # [batch_size, 64]
        
        # Predicción
        prediction = self.predictor(final_state)
        
        return prediction


class TeamExpert(nn.Module):
    """Experto especializado en dinámicas de equipo."""
    
    def __init__(self, config):
        super(TeamExpert, self).__init__()
        
        # Procesador de features de equipo (simplificado)
        self.team_processor = nn.Sequential(
            nn.Linear(config.input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Attention para relaciones de equipo
        self.team_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Predictor (simplificado)
        self.predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del experto de equipo."""
        # Procesar features
        team_features = self.team_processor(x)  # [batch_size, 32]
        
        if team_features.dim() == 2:
            team_features = team_features.unsqueeze(1)  # [batch_size, 1, 32]
        
        # Self-attention para capturar relaciones
        attended, _ = self.team_attention(team_features, team_features, team_features)
        
        # Usar primer elemento
        final_features = attended[:, 0, :]  # [batch_size, 32]
        
        # Predicción
        prediction = self.predictor(final_features)
        
        return prediction


class IndividualExpert(nn.Module):
    """Experto especializado en características individuales."""
    
    def __init__(self, config):
        super(IndividualExpert, self).__init__()
        
        # Red profunda para características individuales (simplificada)
        self.individual_net = nn.Sequential(
            nn.Linear(config.input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del experto individual."""
        if x.dim() > 2:
            # Si es secuencial, usar último timestep
            x = x[:, -1, :] if x.dim() == 3 else x.view(x.size(0), -1)
        
        return self.individual_net(x)


class MatchupExpert(nn.Module):
    """Experto especializado en matchups específicos."""
    
    def __init__(self, config):
        super(MatchupExpert, self).__init__()
        
        # LSTM para capturar patrones de matchup
        self.matchup_lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Attention para matchups importantes
        self.matchup_attention = nn.MultiheadAttention(
            embed_dim=128,  # 64 * 2 (bidirectional)
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Predictor (simplificado)
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del experto de matchup."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Agregar dimensión temporal
        
        # LSTM processing
        lstm_out, _ = self.matchup_lstm(x)  # [batch_size, seq_len, 128]
        
        # Self-attention
        attended, _ = self.matchup_attention(lstm_out, lstm_out, lstm_out)
        
        # Usar último timestep
        final_features = attended[:, -1, :]  # [batch_size, 128]
        
        # Predicción
        prediction = self.predictor(final_features)
        
        return prediction


class HierarchicalEnsemble(SpecializedEnsemble):
    """
    Ensemble jerárquico que combina expertos en múltiples niveles.
    
    Nivel 1: Expertos especializados
    Nivel 2: Meta-expertos que combinan grupos de expertos
    Nivel 3: Super-meta-learner final
    """
    
    def __init__(self, config):
        super(HierarchicalEnsemble, self).__init__(config)
        
        # Meta-expertos de nivel 2
        self.meta_experts = nn.ModuleDict({
            'temporal_meta': self._create_meta_expert([0, 3]),  # Temporal + Matchup
            'spatial_meta': self._create_meta_expert([1, 2])    # Team + Individual
        })
        
        # Super-meta-learner de nivel 3
        self.super_meta = nn.Sequential(
            MLPBlock(2, 16, dropout=0.1),  # 2 meta-expertos
            nn.Linear(16, 1)
        )
    
    def _create_meta_expert(self, expert_indices: List[int]) -> nn.Module:
        """Crea un meta-experto para un grupo de expertos."""
        return nn.Sequential(
            MLPBlock(len(expert_indices), 8, dropout=0.1),
            nn.Linear(8, 1)
        )
    
    def forward(self, x: torch.Tensor, 
                additional_data: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass jerárquico."""
        
        # Nivel 1: Predicciones de expertos base
        expert_predictions = []
        for expert_name, expert in self.experts.items():
            try:
                if "team" in expert_name and additional_data and "graph_data" in additional_data:
                    pred = expert(x, additional_data["graph_data"])
                else:
                    pred = expert(x)
                expert_predictions.append(pred)
            except:
                expert_predictions.append(torch.zeros(x.size(0), 1, device=x.device))
        
        # Nivel 2: Meta-expertos
        temporal_input = torch.cat([expert_predictions[0], expert_predictions[3]], dim=1)
        spatial_input = torch.cat([expert_predictions[1], expert_predictions[2]], dim=1)
        
        temporal_meta_pred = self.meta_experts['temporal_meta'](temporal_input)
        spatial_meta_pred = self.meta_experts['spatial_meta'](spatial_input)
        
        # Nivel 3: Super-meta-learner
        meta_input = torch.cat([temporal_meta_pred, spatial_meta_pred], dim=1)
        final_prediction = self.super_meta(meta_input)
        
        return final_prediction 