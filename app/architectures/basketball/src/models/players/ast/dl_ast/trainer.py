"""
Trainer Unificado para Modelos de Deep Learning AST
==================================================

Maneja el entrenamiento, validación y evaluación de todos los modelos
de Deep Learning especializados en predicción de asistencias.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from pathlib import Path
import json

from .base import BaseDLModel
from .transformer import BasketballTransformer, MultiScaleTransformer
from .lstm_attention import BiLSTMAttention, HierarchicalLSTM, ConvLSTM
from .gnn import PlayerTeamGNN, HierarchicalGNN
from .vae import BasketballVAE, ConditionalVAE, BetaVAE, SequentialVAE
from .ensemble import SpecializedEnsemble, HierarchicalEnsemble
from .hybrid import HybridASTPredictor, MultiScaleHybrid, AdaptiveHybrid

logger = logging.getLogger(__name__)


class DLTrainer:
    """
    Trainer unificado para modelos de Deep Learning AST.
    
    Características:
    - Soporte para todos los tipos de modelos implementados
    - Entrenamiento con validación cruzada
    - Early stopping y regularización
    - Métricas especializadas para AST
    - Guardado y carga de modelos
    - Visualización de resultados
    """
    
    def __init__(self, config, model_type: str = "transformer"):
        """
        Inicializa el trainer.
        
        Args:
            config: Configuración específica del modelo (BaseConfig o subclase)
            model_type: Tipo de modelo a entrenar
        """
        self.config = config
        self.model_type = model_type
        self.model_config = config  # config ya es la configuración específica
        
        # Configuración de dispositivo
        self.device = torch.device(self.model_config.device)
        
        # Escaladores para normalización
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Historial de entrenamiento
        self.training_history = {}
        
        # Modelo actual
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"DLTrainer inicializado: modelo={model_type}, dispositivo={self.device}")
    
    def create_model(self) -> BaseDLModel:
        """Crea el modelo según el tipo especificado."""
        
        model_map = {
            "transformer": BasketballTransformer,
            "multiscale_transformer": MultiScaleTransformer,
            "lstm": BiLSTMAttention,
            "hierarchical_lstm": HierarchicalLSTM,
            "conv_lstm": ConvLSTM,
            "gnn": PlayerTeamGNN,
            "hierarchical_gnn": HierarchicalGNN,
            "vae": BasketballVAE,
            "conditional_vae": ConditionalVAE,
            "beta_vae": BetaVAE,
            "sequential_vae": SequentialVAE,
            "ensemble": SpecializedEnsemble,
            "hierarchical_ensemble": HierarchicalEnsemble,
            "hybrid": HybridASTPredictor,
            "multiscale_hybrid": MultiScaleHybrid,
            "adaptive_hybrid": AdaptiveHybrid
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        model_class = model_map[self.model_type]
        
        # Crear modelo con configuración específica
        if "vae" in self.model_type and self.model_type != "vae":
            # Modelos VAE especiales pueden necesitar parámetros adicionales
            if self.model_type == "conditional_vae":
                model = model_class(self.model_config, condition_dim=10)
            elif self.model_type == "beta_vae":
                model = model_class(self.model_config, beta=2.0)
            else:
                model = model_class(self.model_config)
        else:
            model = model_class(self.model_config)
        
        return model.to(self.device)
    
    def prepare_data(self, df: pd.DataFrame, 
                    sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepara los datos para entrenamiento.
        
        Args:
            df: DataFrame con datos de jugadores
            sequence_length: Longitud de secuencia para modelos temporales
            
        Returns:
            Tuple de (features, targets)
        """
        # Filtrar columnas de features (excluir target y metadatos)
        feature_cols = [col for col in df.columns 
                       if col not in ['assists', 'player', 'date', 'team', 'Opp']]
        
        # Filtrar solo columnas numéricas
        numeric_cols = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
        
        logger.info(f"Features numéricas seleccionadas: {len(numeric_cols)} de {len(feature_cols)} totales")
        
        # Actualizar configuración con el número real de features
        self.model_config.input_features = len(numeric_cols)
        logger.info(f"Configuración actualizada: input_features={self.model_config.input_features}")
        
        # Extraer features y target
        X = df[numeric_cols].values
        y = df['assists'].values.reshape(-1, 1)
        
        # Manejar valores NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Verificar que no hay NaN en target
        valid_mask = ~np.isnan(y.flatten())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Datos después de limpieza: {X.shape[0]} registros válidos")
        
        # Normalizar features
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Crear secuencias si es necesario
        if sequence_length and sequence_length > 1:
            X_sequences, y_sequences = self._create_sequences(
                X_scaled, y_scaled, sequence_length, df['Player'].values
            )
            X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
            y_tensor = torch.tensor(y_sequences, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        
        logger.info(f"Datos preparados: X_shape={X_tensor.shape}, y_shape={y_tensor.shape}")
        
        return X_tensor, y_tensor
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int, players: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crea secuencias temporales por jugador."""
        
        sequences_X = []
        sequences_y = []
        
        # Agrupar por jugador
        unique_players = np.unique(players)
        
        for player in unique_players:
            player_mask = players == player
            player_X = X[player_mask]
            player_y = y[player_mask]
            
            # Crear secuencias para este jugador
            for i in range(sequence_length, len(player_X)):
                seq_X = player_X[i-sequence_length:i]
                seq_y = player_y[i]
                
                sequences_X.append(seq_X)
                sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def train(self, df: pd.DataFrame, 
             validation_split: float = 0.2,
             save_path: Optional[str] = None) -> Dict:
        """
        Entrena el modelo.
        
        Args:
            df: DataFrame con datos de entrenamiento
            validation_split: Proporción de datos para validación
            save_path: Ruta donde guardar el modelo entrenado
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info(f"Iniciando entrenamiento del modelo {self.model_type}")
        
        # Crear modelo
        self.model = self.create_model()
        
        # Preparar datos
        sequence_length = getattr(self.model_config, 'sequence_length', None)
        X, y = self.prepare_data(df, sequence_length)
        
        # División train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.model_config.random_seed
        )
        
        # Crear DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.model_config.batch_size, 
            shuffle=False
        )
        
        # Configurar optimizador
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        
        # Configurar scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.model_config.num_epochs):
            # Entrenamiento
            train_metrics = self._train_epoch(train_loader)
            
            # Validación
            val_metrics = self._validate_epoch(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Guardar mejor modelo
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
            
            # Log progreso
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: "
                           f"train_loss={train_metrics['loss']:.4f}, "
                           f"val_loss={val_metrics['loss']:.4f}, "
                           f"val_mae={val_metrics['mae']:.4f}")
            
            # Early stopping
            if patience_counter >= self.model_config.patience:
                logger.info(f"Early stopping en epoch {epoch}")
                break
        
        # Evaluación final
        final_metrics = self._evaluate_model(val_loader)
        
        logger.info(f"Entrenamiento completado. MAE final: {final_metrics['mae']:.4f}")
        
        return {
            'model_type': self.model_type,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'model_summary': self.model.get_model_summary()
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Entrena una época."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                # VAE models return multiple outputs
                predictions, reconstruction, mu, logvar = self.model(data)
                loss_dict = self.model.compute_loss(predictions, target, reconstruction, data, mu, logvar)
                loss = loss_dict['total_loss']
            elif self.model_type in ["gnn", "hierarchical_gnn"]:
                # GNN models need special handling for graph structure
                predictions = self._forward_gnn(data)
                loss = self.model.compute_loss(predictions, target)
            else:
                # Standard models
                predictions = self.model(data)
                loss = self.model.compute_loss(predictions, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config.gradient_clip)
            
            self.optimizer.step()
            
            # Acumular métricas
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
        
        # Calcular métricas
        avg_loss = total_loss / total_samples
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = self.model.compute_metrics(predictions_tensor, targets_tensor)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Valida una época."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                    predictions, reconstruction, mu, logvar = self.model(data)
                    loss_dict = self.model.compute_loss(predictions, target, reconstruction, data, mu, logvar)
                    loss = loss_dict['total_loss']
                elif self.model_type in ["gnn", "hierarchical_gnn"]:
                    # GNN models need special handling for graph structure
                    predictions = self._forward_gnn(data)
                    loss = self.model.compute_loss(predictions, target)
                else:
                    predictions = self.model(data)
                    loss = self.model.compute_loss(predictions, target)
                
                # Acumular métricas
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calcular métricas
        avg_loss = total_loss / total_samples
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = self.model.compute_metrics(predictions_tensor, targets_tensor)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluación completa del modelo."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                    predictions, _, _, _ = self.model(data)
                elif self.model_type in ["gnn", "hierarchical_gnn"]:
                    predictions = self._forward_gnn(data)
                else:
                    predictions = self.model(data)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Desnormalizar predicciones
        predictions_denorm = self.target_scaler.inverse_transform(
            np.array(all_predictions).reshape(-1, 1)
        ).flatten()
        targets_denorm = self.target_scaler.inverse_transform(
            np.array(all_targets).reshape(-1, 1)
        ).flatten()
        
        # Calcular métricas en escala original
        predictions_tensor = torch.tensor(predictions_denorm)
        targets_tensor = torch.tensor(targets_denorm)
        
        metrics = self.model.compute_metrics(predictions_tensor, targets_tensor)
        
        return metrics
    
    def cross_validate(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict:
        """
        Realiza validación cruzada.
        
        Args:
            df: DataFrame con datos
            cv_folds: Número de folds para CV
            
        Returns:
            Resultados de validación cruzada
        """
        logger.info(f"Iniciando validación cruzada con {cv_folds} folds")
        
        # Preparar datos
        sequence_length = getattr(self.model_config, 'sequence_length', None)
        X, y = self.prepare_data(df, sequence_length)
        
        # Configurar CV
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.model_config.random_seed)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Entrenando fold {fold + 1}/{cv_folds}")
            
            # Crear modelo para este fold
            model = self.create_model()
            
            # Dividir datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Crear DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=self.model_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.model_config.batch_size, shuffle=False)
            
            # Entrenar modelo
            optimizer = optim.AdamW(model.parameters(), lr=self.model_config.learning_rate)
            
            # Entrenamiento simplificado para CV
            for epoch in range(min(50, self.model_config.num_epochs)):  # Menos epochs para CV
                model.train()
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                        predictions, reconstruction, mu, logvar = model(data)
                        loss_dict = model.compute_loss(predictions, target, reconstruction, data, mu, logvar)
                        loss = loss_dict['total_loss']
                    elif self.model_type in ["gnn", "hierarchical_gnn"]:
                        predictions = self._forward_gnn_with_model(model, data)
                        loss = model.compute_loss(predictions, target)
                    else:
                        predictions = model(data)
                        loss = model.compute_loss(predictions, target)
                    
                    loss.backward()
                    optimizer.step()
            
            # Evaluar fold
            model.eval()
            fold_predictions = []
            fold_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                        predictions, _, _, _ = model(data)
                    elif self.model_type in ["gnn", "hierarchical_gnn"]:
                        predictions = self._forward_gnn_with_model(model, data)
                    else:
                        predictions = model(data)
                    
                    fold_predictions.extend(predictions.cpu().numpy())
                    fold_targets.extend(target.cpu().numpy())
            
            # Calcular métricas del fold
            pred_tensor = torch.tensor(fold_predictions)
            target_tensor = torch.tensor(fold_targets)
            fold_metrics = model.compute_metrics(pred_tensor, target_tensor)
            
            cv_results.append(fold_metrics)
            
            logger.info(f"Fold {fold + 1} completado: MAE={fold_metrics['mae']:.4f}")
        
        # Agregar resultados
        cv_summary = {}
        for metric in cv_results[0].keys():
            values = [result[metric] for result in cv_results]
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)
        
        logger.info(f"CV completada: MAE_mean={cv_summary['mae_mean']:.4f} ± {cv_summary['mae_std']:.4f}")
        
        return {
            'cv_results': cv_results,
            'cv_summary': cv_summary,
            'model_type': self.model_type
        }
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado."""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.model_config,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        # logger.info(f"Modelo guardado en: {filepath}")  # Comentado para reducir verbosidad
    
    def load_model(self, filepath: str):
        """Carga un modelo guardado."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.model_config = checkpoint['config']
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.training_history = checkpoint.get('training_history', {})
        
        # Crear y cargar modelo
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Modelo cargado desde: {filepath}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            Predicciones desnormalizadas
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        self.model.eval()
        
        # Preparar datos
        sequence_length = getattr(self.model_config, 'sequence_length', None)
        X, _ = self.prepare_data(df, sequence_length)
        
        # Crear DataLoader
        dataset = TensorDataset(X, torch.zeros(X.size(0), 1))  # Dummy targets
        loader = DataLoader(dataset, batch_size=self.model_config.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                
                if self.model_type in ["vae", "conditional_vae", "beta_vae", "sequential_vae"]:
                    pred, _, _, _ = self.model(data)
                elif self.model_type in ["gnn", "hierarchical_gnn"]:
                    pred = self._forward_gnn(data)
                else:
                    pred = self.model(data)
                
                predictions.extend(pred.cpu().numpy())
        
        # Desnormalizar
        predictions_denorm = self.target_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions_denorm 
    
    def _forward_gnn(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass específico para modelos GNN.
        
        Args:
            data: Tensor de features [batch_size, seq_len, num_features]
            
        Returns:
            Predicciones del modelo GNN
        """
        # Para GNN, necesitamos crear estructura de grafo desde las features
        batch_size, seq_len, num_features = data.shape
        
        # Reshape para crear nodos: [batch_size * seq_len, num_features]
        node_features = data.view(-1, num_features)
        
        # Crear edge_index básico (conexiones temporales)
        edge_index = self._create_temporal_edges(batch_size, seq_len)
        
        # Crear batch tensor para identificar a qué grafo pertenece cada nodo
        batch = torch.repeat_interleave(torch.arange(batch_size), seq_len).to(self.device)
        
        # Forward pass del GNN
        predictions = self.model(node_features, edge_index, batch=batch)
        
        return predictions
    
    def _forward_gnn_with_model(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass específico para modelos GNN con modelo personalizado.
        
        Args:
            model: Modelo GNN específico
            data: Tensor de features [batch_size, seq_len, num_features]
            
        Returns:
            Predicciones del modelo GNN
        """
        # Para GNN, necesitamos crear estructura de grafo desde las features
        batch_size, seq_len, num_features = data.shape
        
        # Reshape para crear nodos: [batch_size * seq_len, num_features]
        node_features = data.view(-1, num_features)
        
        # Crear edge_index básico (conexiones temporales)
        edge_index = self._create_temporal_edges(batch_size, seq_len)
        
        # Crear batch tensor para identificar a qué grafo pertenece cada nodo
        batch = torch.repeat_interleave(torch.arange(batch_size), seq_len).to(self.device)
        
        # Forward pass del GNN
        predictions = model(node_features, edge_index, batch=batch)
        
        return predictions
    
    def _create_temporal_edges(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Crea conexiones temporales para el grafo.
        
        Args:
            batch_size: Tamaño del batch
            seq_len: Longitud de la secuencia
            
        Returns:
            edge_index: [2, num_edges] tensor con conexiones del grafo
        """
        edges = []
        
        for batch_idx in range(batch_size):
            offset = batch_idx * seq_len
            
            # Conexiones temporales secuenciales: t -> t+1
            for t in range(seq_len - 1):
                edges.append([offset + t, offset + t + 1])
                edges.append([offset + t + 1, offset + t])  # Bidireccional
            
            # Conexiones adicionales para capturar dependencias a largo plazo
            if seq_len > 3:
                # Conexiones t -> t+2
                for t in range(seq_len - 2):
                    edges.append([offset + t, offset + t + 2])
                    edges.append([offset + t + 2, offset + t])
        
        if not edges:
            # Si no hay conexiones, crear una conexión dummy
            edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        
        return edge_index

def main():
    """Función principal para probar el trainer."""
    # Cargar datos de ejemplo
    df = pd.read_csv('data/example_ast_data.csv')
    
    # Crear configuración
    config = (BasketballTransformer)
    
    # Crear trainer
    trainer = DLTrainer(config, model_type="transformer")
    
    # Entrenar y guardar modelo
    trainer.train(df, save_path="models/ast_transformer.pth")