"""
Variational Autoencoder para Predicción de Asistencias
=====================================================

Implementa un VAE especializado que aprende representaciones latentes
de los patrones de juego para mejorar la predicción de asistencias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

from .base import BaseDLModel, MLPBlock, initialize_weights

logger = logging.getLogger(__name__)


class BasketballVAE(BaseDLModel):
    """
    Variational Autoencoder para aprendizaje de representaciones en basketball.
    
    Características:
    - Encoder que mapea features a espacio latente
    - Decoder que reconstruye features originales
    - Predictor que usa representación latente para AST
    - Regularización KL para aprendizaje estructurado
    """
    
    def __init__(self, config):
        """
        Inicializa el modelo VAE.
        
        Args:
            config: Configuración del modelo (VAEConfig)
        """
        super(BasketballVAE, self).__init__(config, "BasketballVAE")
        
        # CORREGIDO: Para datos secuenciales, la dimensión de entrada es seq_len * features
        if hasattr(config, 'sequence_length') and config.sequence_length > 1:
            self.input_dim = config.sequence_length * config.input_features
            self.is_sequential = True
        else:
            self.input_dim = config.input_features
            self.is_sequential = False
            
        self.latent_dim = config.latent_dim
        self.encoder_dims = config.encoder_dims
        self.decoder_dims = config.decoder_dims
        
        # Pesos de pérdida
        self.reconstruction_weight = config.reconstruction_weight
        self.kl_weight = config.kl_weight
        self.prediction_weight = config.prediction_weight
        
        # Construir encoder
        self.encoder = self._build_encoder(config)
        
        # Capas de distribución latente
        self.mu_layer = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        self.logvar_layer = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        
        # Construir decoder
        self.decoder = self._build_decoder(config)
        
        # Predictor de AST usando representación latente
        self.ast_predictor = self._build_ast_predictor(config)
        
        # Inicializar pesos
        self.apply(initialize_weights)
        
        logger.info(f"VAE inicializado: input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
                   f"encoder_dims={self.encoder_dims}, decoder_dims={self.decoder_dims}, "
                   f"sequential={self.is_sequential}")
    
    def _build_encoder(self, config) -> nn.Module:
        """Construye el encoder."""
        layers = []
        
        input_dim = self.input_dim
        for output_dim in self.encoder_dims:
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='relu'
            ))
            input_dim = output_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, config) -> nn.Module:
        """Construye el decoder."""
        layers = []
        
        input_dim = self.latent_dim
        for output_dim in self.decoder_dims:
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='relu'
            ))
            input_dim = output_dim
        
        # Capa final para reconstrucción
        layers.append(nn.Linear(input_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _build_ast_predictor(self, config) -> nn.Module:
        """Construye el predictor de AST."""
        predictor_dims = config.predictor_dims
        layers = []
        
        input_dim = self.latent_dim
        for output_dim in predictor_dims[:-1]:
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
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica la entrada al espacio latente.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            
        Returns:
            Tuple de (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametrización para sampling del espacio latente.
        
        Args:
            mu: Media de la distribución latente
            logvar: Log-varianza de la distribución latente
            
        Returns:
            Sample del espacio latente
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica del espacio latente a la entrada original.
        
        Args:
            z: Representación latente
            
        Returns:
            Reconstrucción de la entrada
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo del VAE.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, features] o [batch_size, features]
            
        Returns:
            Tuple de (ast_prediction, reconstruction, mu, logvar)
        """
        # CORREGIDO: Manejar secuencias temporales
        original_shape = x.shape
        if len(x.shape) == 3:  # [batch_size, seq_len, features]
            batch_size, seq_len, features = x.shape
            # Flatten secuencia temporal: [batch_size, seq_len * features]
            x_flattened = x.view(batch_size, -1)
        else:  # [batch_size, features]
            x_flattened = x
            batch_size = x.shape[0]
        
        # Encoding
        mu, logvar = self.encode(x_flattened)
        
        # Reparametrización
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        reconstruction_flat = self.decoder(z)
        
        # Reshape reconstruction para coincidir con entrada original
        if len(original_shape) == 3:
            reconstruction = reconstruction_flat.view(original_shape)
        else:
            reconstruction = reconstruction_flat
        
        # Predicción de AST
        ast_prediction = self.ast_predictor(z)
        
        return ast_prediction, reconstruction, mu, logvar
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    reconstruction: torch.Tensor, original: torch.Tensor,
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calcula la pérdida completa del VAE.
        
        Args:
            predictions: Predicciones de AST
            targets: Valores objetivo de AST
            reconstruction: Reconstrucción de features
            original: Features originales
            mu: Media de distribución latente
            logvar: Log-varianza de distribución latente
            
        Returns:
            Diccionario con diferentes componentes de pérdida
        """
        # Pérdida de predicción
        prediction_loss = F.mse_loss(predictions, targets)
        
        # Pérdida de reconstrucción
        reconstruction_loss = F.mse_loss(reconstruction, original)
        
        # Pérdida KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original.size(0)  # Normalizar por batch size
        
        # Pérdida total
        total_loss = (self.prediction_weight * prediction_loss +
                     self.reconstruction_weight * reconstruction_loss +
                     self.kl_weight * kl_loss)
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene la representación latente de la entrada.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Representación latente
        """
        self.eval()
        
        with torch.no_grad():
            mu, logvar = self.encode(x)
            # Usar la media como representación determinística
            return mu
    
    def generate_samples(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Genera muestras sintéticas del modelo.
        
        Args:
            num_samples: Número de muestras a generar
            device: Dispositivo donde generar
            
        Returns:
            Muestras generadas
        """
        self.eval()
        
        with torch.no_grad():
            # Sample del prior
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decodificar
            generated = self.decode(z)
            
        return generated


class ConditionalVAE(BasketballVAE):
    """
    VAE Condicional que incorpora información contextual.
    
    Permite generar representaciones condicionadas en:
    - Tipo de jugador (PG, SG, SF, PF, C)
    - Situación de juego (casa/visitante, playoffs, etc.)
    - Oponente específico
    """
    
    def __init__(self, config, condition_dim: int = 10):
        """
        Inicializa el CVAE.
        
        Args:
            config: Configuración del modelo
            condition_dim: Dimensión de la información condicional
        """
        self.condition_dim = condition_dim
        
        # Modificar dimensiones para incluir condiciones
        original_input_features = config.input_features
        original_input_dim = config.input_features
        
        # Para datos secuenciales, ajustar correctamente
        if hasattr(config, 'sequence_length') and config.sequence_length > 1:
            config.input_features += condition_dim  # Condiciones por timestep
        else:
            config.input_features += condition_dim  # Concatenar condiciones
        
        super(ConditionalVAE, self).__init__(config)
        
        # Restaurar dimensión original
        config.input_features = original_input_features
        
        # Embedding para condiciones
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Modificar decoder para incluir condiciones
        self.decoder = self._build_conditional_decoder(config)
        
        logger.info(f"CVAE inicializado con condition_dim={condition_dim}")
    
    def _build_conditional_decoder(self, config) -> nn.Module:
        """Construye decoder condicional."""
        layers = []
        
        # Combinar latent + condition embedding
        input_dim = self.latent_dim + 16  # 16 del condition embedding
        
        for output_dim in self.decoder_dims:
            layers.append(MLPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=config.dropout_rate,
                batch_norm=config.batch_norm,
                activation='relu'
            ))
            input_dim = output_dim
        
        # Capa final
        layers.append(nn.Linear(input_dim, config.input_features))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, 
                conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass del CVAE.
        
        Args:
            x: Features de entrada [batch_size, input_dim]
            conditions: Información condicional [batch_size, condition_dim]
            
        Returns:
            Tuple de (ast_prediction, reconstruction, mu, logvar)
        """
        # Concatenar entrada con condiciones para encoding
        x_cond = torch.cat([x, conditions], dim=1)
        
        # Encoding
        mu, logvar = self.encode(x_cond)
        
        # Reparametrización
        z = self.reparameterize(mu, logvar)
        
        # Embedding de condiciones
        cond_emb = self.condition_embedding(conditions)
        
        # Concatenar latent con condition embedding para decoding
        z_cond = torch.cat([z, cond_emb], dim=1)
        
        # Decoding
        reconstruction = self.decode(z_cond)
        
        # Predicción de AST
        ast_prediction = self.ast_predictor(z)
        
        return ast_prediction, reconstruction, mu, logvar


class BetaVAE(BasketballVAE):
    """
    Beta-VAE que permite controlar el balance entre reconstrucción y regularización.
    
    Útil para aprender representaciones más disentangled (separadas).
    """
    
    def __init__(self, config, beta: float = 1.0):
        """
        Inicializa el Beta-VAE.
        
        Args:
            config: Configuración del modelo
            beta: Factor de peso para la pérdida KL (beta > 1 para más disentanglement)
        """
        super(BetaVAE, self).__init__(config)
        
        self.beta = beta
        
        logger.info(f"Beta-VAE inicializado con beta={beta}")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    reconstruction: torch.Tensor, original: torch.Tensor,
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcula pérdida con factor beta."""
        
        # Pérdida de predicción
        prediction_loss = F.mse_loss(predictions, targets)
        
        # Pérdida de reconstrucción
        reconstruction_loss = F.mse_loss(reconstruction, original)
        
        # Pérdida KL con factor beta
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original.size(0)
        
        # Pérdida total con beta
        total_loss = (self.prediction_weight * prediction_loss +
                     self.reconstruction_weight * reconstruction_loss +
                     self.beta * self.kl_weight * kl_loss)
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'beta_kl_loss': self.beta * kl_loss
        }


class SequentialVAE(BaseDLModel):
    """
    VAE que procesa secuencias temporales de juegos.
    
    Combina VAE con LSTM para modelar dependencias temporales
    en el espacio latente.
    """
    
    def __init__(self, config):
        super(SequentialVAE, self).__init__(config, "SequentialVAE")
        
        self.sequence_length = config.sequence_length
        self.input_dim = config.input_features
        self.latent_dim = config.latent_dim
        self.hidden_dim = 64
        
        # Encoder por timestep
        self.timestep_encoder = nn.Sequential(
            MLPBlock(self.input_dim, 128, dropout=config.dropout_rate),
            MLPBlock(128, 64, dropout=config.dropout_rate)
        )
        
        # LSTM para secuencia
        self.sequence_lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Capas latentes
        lstm_output_dim = self.hidden_dim * 2  # Bidirectional
        self.mu_layer = nn.Linear(lstm_output_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(lstm_output_dim, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            MLPBlock(self.latent_dim, 128, dropout=config.dropout_rate),
            MLPBlock(128, 256, dropout=config.dropout_rate),
            nn.Linear(256, self.sequence_length * self.input_dim)
        )
        
        # Predictor AST
        self.ast_predictor = nn.Sequential(
            MLPBlock(self.latent_dim, 64, dropout=config.dropout_rate),
            MLPBlock(64, 32, dropout=config.dropout_rate),
            nn.Linear(32, 1)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica secuencia temporal.
        
        Args:
            x: Secuencia [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple de (mu, logvar)
        """
        batch_size, seq_len, _ = x.shape
        
        # Procesar cada timestep
        timestep_features = []
        for t in range(seq_len):
            features = self.timestep_encoder(x[:, t, :])
            timestep_features.append(features)
        
        # Stack y procesar con LSTM
        sequence_features = torch.stack(timestep_features, dim=1)  # [batch_size, seq_len, 64]
        lstm_output, _ = self.sequence_lstm(sequence_features)
        
        # Usar último estado
        final_state = lstm_output[:, -1, :]  # [batch_size, lstm_output_dim]
        
        # Distribución latente
        mu = self.mu_layer(final_state)
        logvar = self.logvar_layer(final_state)
        
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica a secuencia temporal."""
        batch_size = z.size(0)
        
        # Decodificar
        decoded = self.decoder(z)  # [batch_size, seq_len * input_dim]
        
        # Reshape a secuencia
        decoded = decoded.view(batch_size, self.sequence_length, self.input_dim)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass del Sequential VAE."""
        
        # Encoding
        mu, logvar = self.encode(x)
        
        # Reparametrización
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        reconstruction = self.decode(z)
        
        # Predicción AST
        ast_prediction = self.ast_predictor(z)
        
        return ast_prediction, reconstruction, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrización estándar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std 