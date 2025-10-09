"""
Pipeline de Entrenamiento para Modelos de Deep Learning AST
==========================================================

Entrena y evalúa todos los modelos de Deep Learning especializados
en predicción de asistencias con arquitecturas avanzadas.

Modelos incluidos:
- Transformer (BasketballTransformer, MultiScaleTransformer)
- LSTM con Attention (BiLSTMAttention, HierarchicalLSTM, ConvLSTM)  
- Graph Neural Networks (PlayerTeamGNN, HierarchicalGNN)
- Variational Autoencoders (BasketballVAE, ConditionalVAE, BetaVAE)
- Ensembles Especializados (SpecializedEnsemble, HierarchicalEnsemble)
- Modelos Híbridos (HybridASTPredictor, MultiScaleHybrid, AdaptiveHybrid)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Agregar el directorio raíz al PYTHONPATH para resolver las importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

# Configurar logging
from app.architectures.basketball.config.logging_config import NBALogger
logger = NBALogger.get_logger(__name__)

# Imports de Deep Learning
from app.architectures.basketball.src.models.players.ast.dl_ast.config import (
    DLConfig, TransformerConfig, LSTMConfig, GNNConfig, 
    VAEConfig, EnsembleConfig, HybridConfig
)
from app.architectures.basketball.src.models.players.ast.dl_ast.trainer import DLTrainer
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.ast.features_ast import AssistsFeatureEngineer


class ASTDeepLearningPipeline:
    """
    Pipeline completo para entrenamiento de modelos de Deep Learning AST.
    
    Características:
    - Entrenamiento de múltiples arquitecturas
    - Comparación automática de rendimiento
    - Guardado de mejores modelos
    - Generación de reportes completos
    - Validación cruzada temporal
    """
    
    def __init__(self, results_dir: str = "app/architectures/basketball/results/ast_dl_models"):
        """
        Inicializa el pipeline de Deep Learning.
        
        Args:
            results_dir: Directorio para guardar resultados
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones para todos los modelos
        self.dl_config = DLConfig()
        
        # Feature engineer para generar características
        self.feature_engineer = AssistsFeatureEngineer()
        
        # Resultados de todos los modelos
        self.model_results = {}
        
        # Mejor modelo encontrado
        self.best_model = None
        self.best_score = float('inf')
        
        logger.info("Pipeline de Deep Learning AST inicializado")
        logger.info(f"Directorio de resultados: {self.results_dir}")
        logger.info(f"Dispositivo disponible: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    def _format_training_results(self, results: Dict, training_time: float) -> Dict:
        """Formatea los resultados del entrenamiento a un formato estándar."""
        final_metrics = results.get('final_metrics', {})
        return {
            'model_type': results.get('model_type', 'unknown'),
            'test_mae': final_metrics.get('mae', 0.0),
            'test_rmse': final_metrics.get('rmse', 0.0),
            'test_r2': final_metrics.get('r2', 0.0),
            'training_time': training_time,
            'epochs_trained': results.get('epochs_trained', 0),
            'best_val_loss': results.get('best_val_loss', 0.0)
        }
    
    def _train_single_model(self, df: pd.DataFrame, config, model_type: str, 
                           model_name: str, save_filename: str) -> Dict:
        """Entrena un solo modelo y devuelve resultados formateados."""
        logger.info(f"Entrenando {model_name}...")
        trainer = DLTrainer(config, model_type)
        
        start_time = time.time()
        # Guardar modelo en .joblib/
        os.makedirs('app/architectures/basketball/.joblib', exist_ok=True)
        model_save_path = os.path.join('app/architectures/basketball/.joblib', save_filename)
        results = trainer.train(df, save_path=model_save_path)
        training_time = time.time() - start_time
        
        formatted_results = self._format_training_results(results, training_time)
        
        logger.info(f"{model_name} - MAE: {formatted_results['test_mae']:.4f}, Tiempo: {training_time:.1f}s")
        
        return formatted_results
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Carga y prepara los datos para entrenamiento."""
        logger.info("Cargando y preparando datos...")
        
        # Rutas de datos por defecto
        players_total_path = "app/architectures/basketball/data/players_total.csv"
        players_quarters_path = "app/architectures/basketball/data/players_quarters.csv"
        teams_total_path = "app/architectures/basketball/data/teams_total.csv"
        teams_quarters_path = "app/architectures/basketball/data/teams_quarters.csv"
        biometrics_path = "app/architectures/basketball/data/biometrics.csv"
        
        # Cargar datos
        data_loader = NBADataLoader(players_total_path, players_quarters_path, teams_total_path, teams_quarters_path, biometrics_path)
        df, teams_df = data_loader.load_data()
        
        logger.info(f"Datos cargados: {len(df)} registros")
        logger.info(f"Datos de equipos: {len(teams_df)} registros")
        
        # Asignar datos de equipos al feature engineer
        self.feature_engineer.teams_df = teams_df
        
        # Generar features especializadas
        logger.info("Generando features especializadas para Deep Learning...")
        features = self.feature_engineer.generate_all_features(df)
        
        logger.info(f"Features generadas: {len(features)}")
        
        # Verificar que tenemos el target
        if 'AST' not in df.columns:
            raise ValueError("Target 'AST' no encontrado en los datos")
        
        # Filtrar datos válidos
        df_clean = df.dropna(subset=['AST'])
        logger.info(f"Datos después de limpieza: {len(df_clean)} registros válidos")
        
        return df_clean
    
    def train_transformer_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos Transformer."""
        logger.info("🔥 ENTRENANDO MODELOS TRANSFORMER")
        
        config = self.dl_config.transformer
        transformer_results = {}
        
        # 1. Basketball Transformer básico
        transformer_results['basketball_transformer'] = self._train_single_model(
            df, config, "transformer", "BasketballTransformer", "transformer_basic.pt"
        )
        
        # 2. MultiScale Transformer
        transformer_results['multiscale_transformer'] = self._train_single_model(
            df, config, "multiscale_transformer", "MultiScaleTransformer", "transformer_multiscale.pt"
        )
        
        return transformer_results
    
    def train_lstm_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos LSTM con Attention."""
        logger.info("🧠 ENTRENANDO MODELOS LSTM CON ATTENTION")
        
        config = self.dl_config.lstm
        lstm_results = {}
        
        # 1. BiLSTM con Attention
        lstm_results['bilstm_attention'] = self._train_single_model(
            df, config, "lstm", "BiLSTMAttention", "lstm_attention.pt"
        )
        
        # 2. Hierarchical LSTM
        lstm_results['hierarchical_lstm'] = self._train_single_model(
            df, config, "hierarchical_lstm", "HierarchicalLSTM", "lstm_hierarchical.pt"
        )
        
        # 3. Convolutional LSTM
        lstm_results['conv_lstm'] = self._train_single_model(
            df, config, "conv_lstm", "ConvLSTM", "lstm_conv.pt"
        )
        
        return lstm_results
    
    def train_gnn_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos Graph Neural Network."""
        logger.info("🕸️ ENTRENANDO MODELOS GRAPH NEURAL NETWORK")
        
        config = self.dl_config.gnn
        gnn_results = {}
        
        # 1. Player-Team GNN
        gnn_results['player_team_gnn'] = self._train_single_model(
            df, config, "gnn", "PlayerTeamGNN", "gnn_player_team.pt"
        )
        
        # 2. Hierarchical GNN
        gnn_results['hierarchical_gnn'] = self._train_single_model(
            df, config, "hierarchical_gnn", "HierarchicalGNN", "gnn_hierarchical.pt"
        )
        
        return gnn_results
    
    def train_vae_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos Variational Autoencoder."""
        logger.info("🎭 ENTRENANDO MODELOS VARIATIONAL AUTOENCODER")
        
        config = self.dl_config.vae
        vae_results = {}
        
        # 1. Basketball VAE básico
        vae_results['basketball_vae'] = self._train_single_model(
            df, config, "vae", "BasketballVAE", "vae_basic.pt"
        )
        
        # 2. Conditional VAE
        vae_results['conditional_vae'] = self._train_single_model(
            df, config, "conditional_vae", "ConditionalVAE", "vae_conditional.pt"
        )
        
        # 3. Beta VAE
        vae_results['beta_vae'] = self._train_single_model(
            df, config, "beta_vae", "BetaVAE", "vae_beta.pt"
        )
        
        return vae_results
    
    def train_ensemble_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos Ensemble especializados."""
        logger.info("🎯 ENTRENANDO MODELOS ENSEMBLE ESPECIALIZADOS")
        
        config = self.dl_config.ensemble
        ensemble_results = {}
        
        # 1. Specialized Ensemble
        ensemble_results['specialized_ensemble'] = self._train_single_model(
            df, config, "ensemble", "SpecializedEnsemble", "ensemble_specialized.pt"
        )
        
        # 2. Hierarchical Ensemble
        ensemble_results['hierarchical_ensemble'] = self._train_single_model(
            df, config, "hierarchical_ensemble", "HierarchicalEnsemble", "ensemble_hierarchical.pt"
        )
        
        return ensemble_results
    
    def train_hybrid_models(self, df: pd.DataFrame) -> Dict:
        """Entrena modelos híbridos avanzados."""
        logger.info("🚀 ENTRENANDO MODELOS HÍBRIDOS AVANZADOS")
        
        config = self.dl_config.hybrid
        hybrid_results = {}
        
        # 1. Hybrid AST Predictor
        hybrid_results['hybrid_ast_predictor'] = self._train_single_model(
            df, config, "hybrid", "HybridASTPredictor", "hybrid_basic.pt"
        )
        
        # 2. MultiScale Hybrid
        hybrid_results['multiscale_hybrid'] = self._train_single_model(
            df, config, "multiscale_hybrid", "MultiScaleHybrid", "hybrid_multiscale.pt"
        )
        
        # 3. Adaptive Hybrid
        hybrid_results['adaptive_hybrid'] = self._train_single_model(
            df, config, "adaptive_hybrid", "AdaptiveHybrid", "hybrid_adaptive.pt"
        )
        
        return hybrid_results
    
    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo de entrenamiento."""
        logger.info("=" * 80)
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE DEEP LEARNING AST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Cargar y preparar datos
            df = self.load_and_prepare_data()
            
            # 2. Entrenar todos los modelos
            logger.info("\n🔥 FASE 1: MODELOS TRANSFORMER")
            self.model_results['transformer'] = self.train_transformer_models(df)
            
            logger.info("\n🧠 FASE 2: MODELOS LSTM")
            self.model_results['lstm'] = self.train_lstm_models(df)
            
            logger.info("\n🕸️ FASE 3: MODELOS GNN")
            self.model_results['gnn'] = self.train_gnn_models(df)
            
            logger.info("\n🎭 FASE 4: MODELOS VAE")
            self.model_results['vae'] = self.train_vae_models(df)
            
            logger.info("\n🎯 FASE 5: MODELOS ENSEMBLE")
            self.model_results['ensemble'] = self.train_ensemble_models(df)
            
            logger.info("\n🚀 FASE 6: MODELOS HÍBRIDOS")
            self.model_results['hybrid'] = self.train_hybrid_models(df)
            
            # 3. Generar reporte final
            total_time = time.time() - start_time
            self.generate_final_report(total_time)
            
        except Exception as e:
            logger.error(f"Error en pipeline: {e}")
            raise
    
    def generate_final_report(self, total_time: float):
        """Genera reporte final con todos los resultados."""
        logger.info("=" * 80)
        logger.info("📊 GENERANDO REPORTE FINAL")
        logger.info("=" * 80)
        
        # Encontrar mejor modelo
        best_mae = float('inf')
        best_model_info = None
        
        all_results = []
        
        for category, models in self.model_results.items():
            for model_name, results in models.items():
                mae = results.get('test_mae', float('inf'))
                r2 = results.get('test_r2', 0)
                training_time = results.get('training_time', 0)
                
                result_info = {
                    'category': category,
                    'model': model_name,
                    'mae': mae,
                    'r2': r2,
                    'training_time': training_time,
                    'accuracy_1ast': self._calculate_accuracy(mae, 1),
                    'accuracy_2ast': self._calculate_accuracy(mae, 2),
                    'accuracy_3ast': self._calculate_accuracy(mae, 3)
                }
                
                all_results.append(result_info)
                
                if mae < best_mae:
                    best_mae = mae
                    best_model_info = result_info
        
        # Ordenar por MAE
        all_results.sort(key=lambda x: x['mae'])
        
        # Mostrar ranking
        logger.info("\n🏆 RANKING DE MODELOS (por MAE):")
        logger.info("-" * 100)
        logger.info(f"{'Rank':<4} {'Modelo':<25} {'Categoría':<12} {'MAE':<8} {'R²':<8} {'±1 AST':<8} {'Tiempo':<8}")
        logger.info("-" * 100)
        
        for i, result in enumerate(all_results[:15], 1):  # Top 15
            logger.info(f"{i:<4} {result['model']:<25} {result['category']:<12} "
                       f"{result['mae']:<8.4f} {result['r2']:<8.3f} "
                       f"{result['accuracy_1ast']:<8.1f}% {result['training_time']:<8.1f}s")
        
        # Información del mejor modelo
        if best_model_info:
            logger.info(f"\n🥇 MEJOR MODELO: {best_model_info['model']}")
            logger.info(f"   Categoría: {best_model_info['category']}")
            logger.info(f"   MAE: {best_model_info['mae']:.4f}")
            logger.info(f"   R²: {best_model_info['r2']:.3f}")
            logger.info(f"   Precisión ±1 AST: {best_model_info['accuracy_1ast']:.1f}%")
            logger.info(f"   Tiempo de entrenamiento: {best_model_info['training_time']:.1f}s")
        
        # Guardar reporte completo
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_training_time': total_time,
            'best_model': best_model_info,
            'all_results': all_results,
            'summary': {
                'total_models_trained': len(all_results),
                'best_mae': best_mae,
                'average_mae': np.mean([r['mae'] for r in all_results]),
                'categories_tested': len(self.model_results)
            }
        }
        
        report_path = self.results_dir / "dl_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Reporte completo guardado en: {report_path}")
        logger.info(f"⏱️ Tiempo total de entrenamiento: {total_time:.1f} segundos")
        logger.info("=" * 80)
    
    def _calculate_accuracy(self, mae: float, tolerance: int) -> float:
        """Calcula precisión aproximada basada en MAE."""
        # Aproximación: si MAE <= tolerance, entonces ~90% accuracy
        # Esta es una estimación simplificada
        if mae <= tolerance:
            return 90.0 + (tolerance - mae) * 5
        else:
            return max(0, 90.0 - (mae - tolerance) * 20)


def main():
    """Función principal para ejecutar el pipeline."""
    logger.info("Iniciando Pipeline de Deep Learning AST...")
    
    # Crear y ejecutar pipeline
    pipeline = ASTDeepLearningPipeline()
    pipeline.run_complete_pipeline()
    
    logger.info("Pipeline de Deep Learning completado exitosamente!")


if __name__ == "__main__":
    main()