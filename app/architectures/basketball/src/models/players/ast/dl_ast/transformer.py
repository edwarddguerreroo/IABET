"""
Modelo Transformer para Predicción de Asistencias
================================================

Implementa un Transformer especializado que captura dependencias temporales
complejas en las secuencias de rendimiento de jugadores para predecir asistencias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

from .base import BaseDLModel, PositionalEncoding, MLPBlock, initialize_weights

logger = logging.getLogger(__name__)


class BasketballTransformer(BaseDLModel):
    """
    Transformer especializado para predicción de asistencias en basketball.
    
    Características:
    - Attention multi-cabeza para capturar patrones temporales complejos
    - Codificación posicional para secuencias temporales
    - Cabezas especializadas para diferentes aspectos del juego
    - Regularización avanzada para prevenir overfitting
    """
    
    def __init__(self, config):
        """
        Inicializa el modelo Transformer.
        
        Args:
            config: Configuración del modelo (TransformerConfig)
        """
        super(BasketballTransformer, self).__init__(config, "BasketballTransformer")
        
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_encoder_layers
        self.sequence_length = config.sequence_length
        
        # Embedding de entrada
        self.input_embedding = nn.Linear(config.input_features, self.d_model)
        
        # Codificación posicional
        self.positional_encoding = PositionalEncoding(
            self.d_model, 
            max_len=config.max_sequence_length
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm para mejor estabilidad
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # Attention pooling para agregación final
        self.attention_pooling = AttentionPooling(self.d_model)
        
        # Cabezas especializadas
        self._build_prediction_heads(config)
        
        # Inicializar pesos
        self.apply(initialize_weights)
        
        logger.info(f"Transformer inicializado: d_model={self.d_model}, "
                   f"nhead={self.nhead}, layers={self.num_layers}")
    
    def _build_prediction_heads(self, config):
        """Construye las cabezas de predicción especializadas."""
        
        # Cabeza principal de AST
        head_dims = config.prediction_head_dims
        layers = []
        
        input_dim = self.d_model
        for i, output_dim in enumerate(head_dims[:-1]):
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='gelu'
            ))
            input_dim = output_dim
        
        # Capa final sin activación
        layers.append(nn.Linear(input_dim, head_dims[-1]))
        
        self.ast_head = nn.Sequential(*layers)
        
        # Cabeza auxiliar para contexto (opcional)
        self.context_head = nn.Sequential(
            MLPBlock(self.d_model, 64, dropout=config.dropout_rate),
            MLPBlock(64, 32, dropout=config.dropout_rate),
            nn.Linear(32, 5)  # Predice contexto: minutos, FG%, etc.
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del Transformer.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, features]
            mask: Máscara de padding opcional [batch_size, seq_len]
            
        Returns:
            Predicciones de asistencias [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Embedding de entrada
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Aplicar codificación posicional
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Crear máscara de atención si es necesaria
        if mask is not None:
            # Convertir máscara de padding a máscara de atención
            attention_mask = mask.bool()
        else:
            attention_mask = None
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(
            x, 
            src_key_padding_mask=attention_mask
        )  # [batch_size, seq_len, d_model]
        
        # Agregación con attention pooling
        pooled_output, attention_weights = self.attention_pooling(
            transformer_output, 
            mask=attention_mask
        )  # [batch_size, d_model]
        
        # Predicción principal
        ast_prediction = self.ast_head(pooled_output)  # [batch_size, 1]
        
        return ast_prediction
    
    def forward_with_attention(self, x: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass que también retorna los pesos de atención.
        
        Args:
            x: Tensor de entrada
            mask: Máscara opcional
            
        Returns:
            Tuple de (predicciones, pesos_de_atención)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embedding y codificación posicional
        x = self.input_embedding(x)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        # Máscara de atención
        attention_mask = mask.bool() if mask is not None else None
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        # Agregación con attention
        pooled_output, attention_weights = self.attention_pooling(
            transformer_output, mask=attention_mask
        )
        
        # Predicción
        ast_prediction = self.ast_head(pooled_output)
        
        return ast_prediction, attention_weights
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida del Transformer con regularización adicional.
        
        Args:
            predictions: Predicciones del modelo
            targets: Valores objetivo
            
        Returns:
            Pérdida total
        """
        # Pérdida base
        base_loss = super().compute_loss(predictions, targets)
        
        # Regularización L2 en las cabezas de predicción
        l2_reg = 0.0
        for param in self.ast_head.parameters():
            l2_reg += torch.norm(param, p=2)
        
        # Pérdida total
        total_loss = base_loss + self.config.weight_decay * l2_reg
        
        return total_loss
    
    def get_attention_patterns(self, x: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> dict:
        """
        Extrae patrones de atención para análisis.
        
        Args:
            x: Tensor de entrada
            mask: Máscara opcional
            
        Returns:
            Diccionario con patrones de atención
        """
        self.eval()
        
        with torch.no_grad():
            predictions, attention_weights = self.forward_with_attention(x, mask)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy(),
            'most_important_games': torch.argmax(attention_weights, dim=1).cpu().numpy()
        }


class AttentionPooling(nn.Module):
    """Pooling especializado con atención para agregación de secuencias temporales."""
    
    def __init__(self, input_dim: int):
        super(AttentionPooling, self).__init__()
        
        # Red de atención más sofisticada
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1)
        )
        
        # Normalización de atención
        self.attention_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del attention pooling.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_dim]
            mask: Máscara de padding [batch_size, seq_len]
            
        Returns:
            Tuple de (output_agregado, pesos_de_atención)
        """
        # Normalizar entrada
        x_norm = self.attention_norm(x)
        
        # Calcular pesos de atención
        attention_scores = self.attention_net(x_norm)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Aplicar máscara si existe
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.bool(), -1e9)
        
        # Softmax para obtener pesos normalizados
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Aplicar atención
        weighted_output = torch.sum(
            x * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, input_dim]
        
        return weighted_output, attention_weights


class MultiScaleTransformer(BasketballTransformer):
    """
    Transformer multi-escala que procesa diferentes ventanas temporales.
    
    Captura patrones tanto a corto plazo (últimos juegos) como a largo plazo
    (tendencias de temporada).
    """
    
    def __init__(self, config):
        super(MultiScaleTransformer, self).__init__(config)
        
        # Escalas temporales diferentes
        self.scales = [3, 5, 10, 20]  # Ventanas de 3, 5, 10, 20 juegos
        
        # Transformers especializados por escala
        self.scale_transformers = nn.ModuleDict()
        
        for scale in self.scales:
            # Transformer más pequeño para cada escala
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model // 2,
                nhead=self.nhead // 2,
                dim_feedforward=config.dim_feedforward // 2,
                dropout=config.transformer_dropout,
                batch_first=True
            )
            
            self.scale_transformers[f'scale_{scale}'] = nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
        
        # Fusión de escalas
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(self.scales) * (self.d_model // 2), self.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.d_model, self.d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass multi-escala."""
        batch_size, seq_len, _ = x.shape
        
        # Embedding inicial
        x_embedded = self.input_embedding(x)
        
        # Procesar cada escala
        scale_outputs = []
        
        for scale in self.scales:
            # Tomar últimos 'scale' elementos
            if seq_len >= scale:
                x_scale = x_embedded[:, -scale:, :]
                
                # Reducir dimensionalidad para esta escala
                x_scale_reduced = nn.Linear(
                    self.d_model, self.d_model // 2
                ).to(x.device)(x_scale)
                
                # Procesar con transformer de esta escala
                scale_output = self.scale_transformers[f'scale_{scale}'](x_scale_reduced)
                
                # Pooling temporal
                scale_pooled = torch.mean(scale_output, dim=1)  # [batch_size, d_model//2]
                scale_outputs.append(scale_pooled)
            else:
                # Si no hay suficientes datos, usar padding
                scale_outputs.append(
                    torch.zeros(batch_size, self.d_model // 2, device=x.device)
                )
        
        # Fusionar escalas
        fused_features = torch.cat(scale_outputs, dim=1)  # [batch_size, len(scales) * d_model//2]
        fused_output = self.scale_fusion(fused_features)  # [batch_size, d_model]
        
        # Predicción final
        ast_prediction = self.ast_head(fused_output)
        
        return ast_prediction 