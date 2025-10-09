#!/usr/bin/env python3
"""
Script de prueba rápida para el modelo PTS
==========================================

Entrena el modelo PTS con solo 50 muestras para verificar que todo funciona
y poder depurar problemas rápidamente.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Cargar una muestra pequeña de datos para prueba rápida"""
    logger.info("Cargando datos de muestra...")
    
    # Cargar datos de jugadores
    players_df = pd.read_csv('app/architectures/basketball/data/players_total.csv')
    logger.info(f"Datos de jugadores cargados: {len(players_df)} registros")
    
    # Cargar datos de equipos
    teams_df = pd.read_csv('app/architectures/basketball/data/teams_total.csv')
    logger.info(f"Datos de equipos cargados: {len(teams_df)} registros")
    
    # Tomar solo 50 muestras de jugadores (mantener orden cronológico)
    players_df['Date'] = pd.to_datetime(players_df['Date'])
    players_df = players_df.sort_values(['player', 'Date'])
    
    # Seleccionar jugadores con más partidos para tener datos suficientes
    player_counts = players_df['player'].value_counts()
    top_players = player_counts.head(5).index.tolist()  # Top 5 jugadores
    
    # Filtrar por top jugadores y tomar primeras 50 muestras
    sample_df = players_df[players_df['player'].isin(top_players)].head(50)
    
    logger.info(f"Muestra seleccionada: {len(sample_df)} registros")
    logger.info(f"Jugadores en muestra: {sample_df['player'].nunique()}")
    logger.info(f"Rango de fechas: {sample_df['Date'].min()} a {sample_df['Date'].max()}")
    
    return sample_df, teams_df

def test_pts_model():
    """Probar el modelo PTS con datos de muestra"""
    try:
        logger.info("=" * 60)
        logger.info("INICIANDO PRUEBA RÁPIDA DEL MODELO PTS")
        logger.info("=" * 60)
        
        # Cargar datos
        players_df, teams_df = load_sample_data()
        
        # Importar el modelo
        from app.architectures.basketball.src.models.players.pts.model_pts import StackingPTSModel
        
        # Crear modelo con configuración mínima
        model = StackingPTSModel(
            n_trials=5,  # Solo 5 trials para prueba rápida
            cv_folds=2,  # Solo 2 folds para prueba rápida
            early_stopping_rounds=10,
            random_state=42
        )
        
        logger.info("Modelo PTS creado exitosamente")
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        results = model.train(players_df)
        
        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"MAE: {results.get('mae', 'N/A'):.4f}")
        logger.info(f"RMSE: {results.get('rmse', 'N/A'):.4f}")
        logger.info(f"R²: {results.get('r2', 'N/A'):.4f}")
        logger.info("=" * 60)
        
        # Probar predicción
        logger.info("Probando predicción...")
        predictions = model.predict(players_df.head(10))
        logger.info(f"Predicciones generadas: {len(predictions)} valores")
        logger.info(f"Rango de predicciones: {predictions.min():.2f} - {predictions.max():.2f}")
        
        # Verificar que el modelo está entrenado
        if hasattr(model, 'is_trained') and model.is_trained:
            logger.info("✅ Modelo marcado como entrenado correctamente")
        else:
            logger.warning("⚠️ Modelo no marcado como entrenado")
        
        # Verificar componentes del modelo
        if hasattr(model, 'trained_base_models') and model.trained_base_models:
            logger.info(f"✅ Modelos base entrenados: {len(model.trained_base_models)}")
        else:
            logger.warning("⚠️ No hay modelos base entrenados")
        
        if hasattr(model, 'meta_learner') and model.meta_learner:
            logger.info("✅ Meta-learner entrenado")
        else:
            logger.warning("⚠️ Meta-learner no entrenado")
        
        logger.info("=" * 60)
        logger.info("PRUEBA COMPLETADA EXITOSAMENTE")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante la prueba: {e}")
        logger.error(f"Tipo de error: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        return False

def main():
    """Función principal"""
    logger.info("Iniciando script de prueba rápida del modelo PTS")
    
    # Cambiar al directorio correcto
    os.chdir(Path(__file__).parent.parent.parent.parent)
    
    # Ejecutar prueba
    success = test_pts_model()
    
    if success:
        logger.info("🎉 ¡Prueba exitosa! El modelo PTS funciona correctamente.")
        return 0
    else:
        logger.error("💥 Prueba fallida. Revisar errores arriba.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
