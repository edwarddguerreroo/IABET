"""
Modelo LSTM Bidireccional con Attention para Predicción de Asistencias
=====================================================================

Implementa un LSTM bidireccional con mecanismo de atención especializado
para capturar patrones temporales y dependencias a largo plazo en asistencias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from .base import BaseDLModel, MLPBlock, initialize_weights

logger = logging.getLogger(__name__)


class BiLSTMAttention(BaseDLModel):
    """
    LSTM Bidireccional con Attention para predicción de asistencias.
    
    Características:
    - LSTM bidireccional para capturar contexto pasado y futuro
    - Self-attention para identificar juegos más relevantes
    - Múltiples cabezas de atención para diferentes aspectos
    - Regularización avanzada con dropout y batch normalization
    """
    
    def __init__(self, config):
        """
        Inicializa el modelo LSTM con Attention.
        
        Args:
            config: Configuración del modelo (LSTMConfig)
        """
        super(BiLSTMAttention, self).__init__(config, "BiLSTMAttention")
        
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.sequence_length = config.sequence_length
        
        # Dimensión efectiva después del LSTM
        self.lstm_output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=config.lstm_dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention multi-cabeza
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Normalización después de attention
        self.attention_norm = nn.LayerNorm(self.lstm_output_dim)
        
        # Attention pooling personalizado
        self.attention_pooling = LSTMAttentionPooling(self.lstm_output_dim)
        
        # Cabezas de predicción
        self._build_prediction_heads(config)
        
        # Inicializar pesos
        self.apply(initialize_weights)
        
        logger.info(f"BiLSTM inicializado: hidden_size={self.hidden_size}, "
                   f"layers={self.num_layers}, bidirectional={self.bidirectional}")
    
    def _build_prediction_heads(self, config):
        """Construye las cabezas de predicción especializadas."""
        
        # Cabeza principal de AST
        predictor_dims = config.predictor_dims
        layers = []
        
        input_dim = self.lstm_output_dim
        for i, output_dim in enumerate(predictor_dims[:-1]):
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='relu'
            ))
            input_dim = output_dim
        
        # Capa final
        layers.append(nn.Linear(input_dim, predictor_dims[-1]))
        
        self.ast_predictor = nn.Sequential(*layers)
        
        # Cabeza auxiliar para tendencias
        self.trend_predictor = nn.Sequential(
            MLPBlock(self.lstm_output_dim, 64, dropout=config.dropout_rate),
            MLPBlock(64, 32, dropout=config.dropout_rate),
            nn.Linear(32, 3)  # Tendencia: subiendo, estable, bajando
        )
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del BiLSTM con Attention.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, features]
            mask: Máscara de padding opcional [batch_size, seq_len]
            
        Returns:
            Predicciones de asistencias [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, lstm_output_dim]
        
        # Self-attention
        attn_output, attn_weights = self.self_attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=mask.bool() if mask is not None else None
        )  # [batch_size, seq_len, lstm_output_dim]
        
        # Residual connection y normalización
        attn_output = self.attention_norm(attn_output + lstm_output)
        
        # Attention pooling para agregación final
        pooled_output, pooling_weights = self.attention_pooling(
            attn_output, mask=mask
        )  # [batch_size, lstm_output_dim]
        
        # Predicción principal
        ast_prediction = self.ast_predictor(pooled_output)  # [batch_size, 1]
        
        return ast_prediction
    
    def forward_with_attention(self, x: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass que retorna predicciones y pesos de atención.
        
        Args:
            x: Tensor de entrada
            mask: Máscara opcional
            
        Returns:
            Tuple de (predicciones, diccionario_de_atención)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        attn_output, self_attn_weights = self.self_attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=mask.bool() if mask is not None else None
        )
        
        # Normalización
        attn_output = self.attention_norm(attn_output + lstm_output)
        
        # Attention pooling
        pooled_output, pooling_weights = self.attention_pooling(attn_output, mask=mask)
        
        # Predicción
        ast_prediction = self.ast_predictor(pooled_output)
        
        # Información de atención
        attention_info = {
            'self_attention_weights': self_attn_weights.detach().cpu().numpy(),
            'pooling_weights': pooling_weights.detach().cpu().numpy(),
            'lstm_hidden_states': hidden.detach().cpu().numpy(),
            'most_important_timesteps': torch.argmax(pooling_weights, dim=1).cpu().numpy()
        }
        
        return ast_prediction, attention_info
    
    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extrae los estados ocultos del LSTM para análisis.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tuple de (lstm_output, final_hidden_state)
        """
        self.eval()
        
        with torch.no_grad():
            lstm_output, (hidden, cell) = self.lstm(x)
        
        return lstm_output, hidden
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida con regularización específica para LSTM.
        
        Args:
            predictions: Predicciones del modelo
            targets: Valores objetivo
            
        Returns:
            Pérdida total
        """
        # Pérdida base
        base_loss = super().compute_loss(predictions, targets)
        
        # Regularización en los pesos del LSTM
        lstm_reg = 0.0
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                lstm_reg += torch.norm(param, p=2)
        
        # Pérdida total
        total_loss = base_loss + self.config.weight_decay * lstm_reg * 0.1
        
        return total_loss


class LSTMAttentionPooling(nn.Module):
    """Pooling especializado con atención para salidas de LSTM."""
    
    def __init__(self, input_dim: int):
        super(LSTMAttentionPooling, self).__init__()
        
        # Red de atención más sofisticada para LSTM
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Normalización
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Proyección adicional
        self.output_projection = nn.Linear(input_dim, input_dim)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del attention pooling para LSTM.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_dim]
            mask: Máscara de padding [batch_size, seq_len]
            
        Returns:
            Tuple de (output_agregado, pesos_de_atención)
        """
        # Normalizar entrada
        x_norm = self.layer_norm(x)
        
        # Calcular scores de atención
        attention_scores = self.attention_net(x_norm)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Aplicar máscara si existe
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.bool(), -1e9)
        
        # Softmax para normalizar
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Aplicar atención
        weighted_output = torch.sum(
            x * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, input_dim]
        
        # Proyección final
        output = self.output_projection(weighted_output)
        
        return output, attention_weights


class HierarchicalLSTM(BiLSTMAttention):
    """
    LSTM Jerárquico que procesa información a múltiples niveles temporales.
    
    Nivel 1: Juego por juego (granularidad fina)
    Nivel 2: Ventanas de 5 juegos (tendencias cortas)
    Nivel 3: Ventanas de 15 juegos (tendencias largas)
    """
    
    def __init__(self, config):
        super(HierarchicalLSTM, self).__init__(config)
        
        # LSTMs para diferentes niveles jerárquicos
        self.game_level_lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        self.short_term_lstm = nn.LSTM(
            input_size=self.hidden_size,  # Output del nivel anterior
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        self.long_term_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusión de niveles - CORREGIDA para coincidir con ast_predictor
        self.level_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size * 2, self.lstm_output_dim)  # Output debe coincidir con ast_predictor
        )
        
        # Attention para cada nivel
        self.level_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass jerárquico."""
        batch_size, seq_len, _ = x.shape
        
        # Nivel 1: Procesamiento juego por juego
        game_output, _ = self.game_level_lstm(x)  # [batch_size, seq_len, hidden_size]
        
        # Nivel 2: Agregación en ventanas de 5 juegos
        if seq_len >= 5:
            # Reshape para ventanas de 5
            window_size = 5
            num_windows = seq_len // window_size
            
            if num_windows > 0:
                # Tomar las últimas ventanas completas
                windowed_input = game_output[:, -num_windows*window_size:, :]
                windowed_input = windowed_input.view(
                    batch_size, num_windows, window_size, self.hidden_size
                )
                
                # Promediar cada ventana
                windowed_features = torch.mean(windowed_input, dim=2)  # [batch_size, num_windows, hidden_size]
                
                # Procesar con LSTM de corto plazo
                short_output, _ = self.short_term_lstm(windowed_features)
            else:
                short_output = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
        else:
            short_output = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
        
        # Nivel 3: Agregación en ventanas de 15 juegos
        if seq_len >= 15:
            window_size = 15
            num_windows = seq_len // window_size
            
            if num_windows > 0:
                windowed_input = game_output[:, -num_windows*window_size:, :]
                windowed_input = windowed_input.view(
                    batch_size, num_windows, window_size, self.hidden_size
                )
                windowed_features = torch.mean(windowed_input, dim=2)
                long_output, _ = self.long_term_lstm(windowed_features)
            else:
                long_output = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
        else:
            long_output = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
        
        # Fusionar niveles
        # Tomar último estado de cada nivel
        game_final = game_output[:, -1, :]  # [batch_size, hidden_size]
        short_final = short_output[:, -1, :] if short_output.size(1) > 0 else torch.zeros_like(game_final)
        long_final = long_output[:, -1, :] if long_output.size(1) > 0 else torch.zeros_like(game_final)
        
        # Concatenar y fusionar
        fused_features = torch.cat([game_final, short_final, long_final], dim=1)
        fused_output = self.level_fusion(fused_features)  # [batch_size, lstm_output_dim]
        
        # Predicción final
        ast_prediction = self.ast_predictor(fused_output)
        
        return ast_prediction


class ConvLSTM(BaseDLModel):
    """
    Modelo híbrido que combina CNN para extracción de features locales
    con LSTM para modelado temporal.
    """
    
    def __init__(self, config):
        super(ConvLSTM, self).__init__(config, "ConvLSTM")
        
        # CNN para extracción de features locales
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # LSTM para modelado temporal
        self.lstm = nn.LSTM(
            input_size=64,  # Output de CNN
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.lstm_dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention y predicción
        lstm_output_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.attention_pooling = LSTMAttentionPooling(lstm_output_dim)
        
        self.predictor = nn.Sequential(
            MLPBlock(lstm_output_dim, 128, dropout=config.dropout_rate),
            MLPBlock(128, 64, dropout=config.dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass del ConvLSTM."""
        batch_size, seq_len, features = x.shape
        
        # CNN processing (necesita transponer para Conv1d)
        x_conv = x.transpose(1, 2)  # [batch_size, features, seq_len]
        conv_output = self.conv_layers(x_conv)  # [batch_size, 64, seq_len]
        conv_output = conv_output.transpose(1, 2)  # [batch_size, seq_len, 64]
        
        # LSTM processing
        lstm_output, _ = self.lstm(conv_output)
        
        # Attention pooling
        pooled_output, _ = self.attention_pooling(lstm_output, mask=mask)
        
        # Predicción
        prediction = self.predictor(pooled_output)
        
        return prediction 