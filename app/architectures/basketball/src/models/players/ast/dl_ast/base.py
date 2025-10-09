"""
Clase Base para Modelos de Deep Learning AST
===========================================

Contiene la clase base abstracta que define la interfaz común
para todos los modelos de Deep Learning especializados en AST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class BaseDLModel(nn.Module, ABC):
    """
    Clase base abstracta para todos los modelos de Deep Learning AST.
    
    Define la interfaz común y funcionalidades compartidas entre todos
    los modelos especializados.
    """
    
    def __init__(self, config, model_name: str = "BaseDLModel"):
        """
        Inicializa el modelo base.
        
        Args:
            config: Configuración del modelo
            model_name: Nombre del modelo para logging
        """
        super(BaseDLModel, self).__init__()
        
        self.config = config
        self.model_name = model_name
        self.device = config.device
        self.dtype = config.dtype
        
        # Métricas de entrenamiento
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': []
        }
        
        # Estado del modelo
        self.is_trained = False
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info(f"Inicializado {model_name} en dispositivo: {self.device}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Predicciones del modelo
        """
        pass
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Args:
            x: Datos de entrada
            
        Returns:
            Predicciones como numpy array
        """
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        
        with torch.no_grad():
            predictions = self.forward(x)
            
        return predictions.cpu().numpy()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida del modelo.
        
        Args:
            predictions: Predicciones del modelo
            targets: Valores objetivo
            
        Returns:
            Pérdida calculada
        """
        # Pérdida base: MAE + MSE
        mae_loss = F.l1_loss(predictions, targets)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Combinar pérdidas
        total_loss = mae_loss + 0.5 * mse_loss
        
        return total_loss
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calcula métricas de evaluación.
        
        Args:
            predictions: Predicciones del modelo
            targets: Valores objetivo
            
        Returns:
            Diccionario con métricas
        """
        # Convertir a numpy
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Calcular métricas
        mae = mean_absolute_error(target_np, pred_np)
        mse = mean_squared_error(target_np, pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_np, pred_np)
        
        # Métricas específicas de AST
        accuracy_1 = np.mean(np.abs(pred_np - target_np) <= 1.0) * 100
        accuracy_2 = np.mean(np.abs(pred_np - target_np) <= 2.0) * 100
        accuracy_3 = np.mean(np.abs(pred_np - target_np) <= 3.0) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy_1ast': accuracy_1,
            'accuracy_2ast': accuracy_2,
            'accuracy_3ast': accuracy_3
        }
    
    def update_training_history(self, epoch_metrics: Dict[str, float]):
        """
        Actualiza el historial de entrenamiento.
        
        Args:
            epoch_metrics: Métricas del epoch actual
        """
        for key, value in epoch_metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Verifica si se debe aplicar early stopping.
        
        Args:
            val_loss: Pérdida de validación actual
            
        Returns:
            True si se debe parar el entrenamiento
        """
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.config.patience
    
    def get_model_summary(self) -> Dict:
        """
        Obtiene un resumen del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'model_summary': self.get_model_summary()
        }, filepath)
        
        logger.info(f"Modelo {self.model_name} guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, config=None):
        """
        Carga un modelo guardado.
        
        Args:
            filepath: Ruta del modelo guardado
            config: Configuración (opcional, se usa la guardada si no se proporciona)
            
        Returns:
            Modelo cargado
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if config is None:
            config = checkpoint['config']
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_history = checkpoint.get('training_history', {})
        model.is_trained = True
        
        logger.info(f"Modelo cargado desde: {filepath}")
        
        return model


class MLPBlock(nn.Module):
    """Bloque MLP reutilizable con normalización y dropout."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2, 
                 batch_norm: bool = True, activation: str = 'relu'):
        super(MLPBlock, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Función de activación
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Codificación posicional para Transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class AttentionPooling(nn.Module):
    """Pooling basado en atención para agregación de secuencias."""
    
    def __init__(self, input_dim: int):
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        
        # Calcular pesos de atención
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Aplicar atención
        weighted_output = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        
        return weighted_output, attention_weights.squeeze(-1)


def initialize_weights(module: nn.Module):
    """Inicializa los pesos de un módulo."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0) 