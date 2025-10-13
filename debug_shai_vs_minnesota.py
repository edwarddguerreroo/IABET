#!/usr/bin/env python3
"""
Debug Detallado: Shai vs Minnesota
==================================

Script completo para debuggear la predicción de Shai Gilgeous-Alexander vs Minnesota Timberwolves
con análisis detallado de todos los componentes.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import logging
import json

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Importar modelos y data loaders
from app.architectures.basketball.pipelines.predict.players.pts_predictor import PTSPredictor
from app.architectures.basketball.pipelines.predict.utils_predict.confidence_predict import PlayersConfidence
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader

def debug_shai_vs_minnesota():
    """Debug completo de Shai vs Minnesota"""
    
    print("🔍 DEBUG DETALLADO: SHAI vs MINNESOTA")
    print("=" * 60)
    
    # 1. CARGAR DATOS HISTÓRICOS
    print("\n📊 1. DATOS HISTÓRICOS DE SHAI:")
    print("-" * 40)
    
    confidence = PlayersConfidence()
    confidence.load_data()
    
    # Buscar datos de Shai
    shai_data = confidence.common_utils._smart_player_search(confidence.historical_players, 'Shai Gilgeous-Alexander')
    
    if len(shai_data) > 0:
        print(f"✅ Datos encontrados: {len(shai_data)} juegos")
        print(f"   Promedio general: {shai_data['points'].mean():.1f} pts")
        print(f"   Mediana: {shai_data['points'].median():.1f} pts")
        print(f"   Std: {shai_data['points'].std():.1f} pts")
        print(f"   Min: {shai_data['points'].min():.0f} pts")
        print(f"   Max: {shai_data['points'].max():.0f} pts")
        
        # Últimos juegos
        print(f"\n   Últimos 10 juegos: {shai_data.head(10)['points'].mean():.1f} pts")
        print(f"   Últimos 5 juegos: {shai_data.head(5)['points'].mean():.1f} pts")
        print(f"   Últimos 3 juegos: {shai_data.head(3)['points'].mean():.1f} pts")
        
        # Últimos juegos específicos
        print(f"\n   Últimos 5 juegos específicos:")
        for i, (_, game) in enumerate(shai_data.head(5).iterrows()):
            print(f"     {i+1}. {game['Date']}: {game['points']:.0f} pts vs {game['Opp']}")
    else:
        print("❌ No se encontraron datos de Shai")
        return
    
    # 2. ANÁLISIS H2H vs MINNESOTA
    print("\n🎯 2. ANÁLISIS H2H vs MINNESOTA:")
    print("-" * 40)
    
    h2h_stats = confidence.calculate_player_h2h_stats('Shai Gilgeous-Alexander', 'MIN', 'points')
    
    print(f"✅ Estadísticas H2H vs Minnesota:")
    print(f"   Juegos encontrados: {h2h_stats['games_found']}")
    print(f"   H2H promedio: {h2h_stats['h2h_mean']:.1f} pts")
    print(f"   H2H std: {h2h_stats['h2h_std']:.1f} pts")
    print(f"   H2H min: {h2h_stats['h2h_min']:.0f} pts")
    print(f"   H2H max: {h2h_stats['h2h_max']:.0f} pts")
    print(f"   H2H factor: {h2h_stats['h2h_factor']:.3f}")
    print(f"   Consistency score: {h2h_stats['consistency_score']:.1f}%")
    print(f"   Últimos 5 H2H: {h2h_stats['last_5_mean']:.1f} pts")
    print(f"   Últimos 10 H2H: {h2h_stats['last_10_mean']:.1f} pts")
    
    # Mostrar juegos específicos vs Minnesota
    minnesota_games = shai_data[shai_data['Opp'] == 'MIN'].head(10)
    if len(minnesota_games) > 0:
        print(f"\n   Últimos juegos vs Minnesota:")
        for i, (_, game) in enumerate(minnesota_games.iterrows()):
            print(f"     {i+1}. {game['Date']}: {game['points']:.0f} pts")
    
    # 3. CREAR MOCKUP Y PREDICCIÓN
    print("\n🏀 3. CREAR MOCKUP Y PREDICCIÓN:")
    print("-" * 40)
    
    # Mockup: OKC vs Minnesota
    mock_game = {
        "gameId": "sr:match:debug123",
        "scheduled": "2024-01-15T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Oklahoma City Thunder",
            "alias": "OKC",
            "players": [
                {
                    "playerId": "sr:player:shai",
                    "fullName": "Shai Gilgeous-Alexander",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "2",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Minnesota Timberwolves", 
            "alias": "MIN",
            "players": [
                {
                    "playerId": "sr:player:ant",
                    "fullName": "Anthony Edwards",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "5",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "Paycom Center",
            "capacity": 18203
        }
    }
    
    print("✅ Mockup creado: OKC vs Minnesota")
    
    # 4. EJECUTAR PREDICCIÓN
    print("\n🔮 4. EJECUTAR PREDICCIÓN:")
    print("-" * 40)
    
    predictor = PTSPredictor()
    prediction_result = predictor.predict_game(mock_game, "Shai Gilgeous-Alexander")
    
    if prediction_result is not None:
        print("✅ Predicción exitosa!")
        
        # Análisis detallado de la predicción
        details = prediction_result['prediction_details']
        
        print(f"\n📈 ANÁLISIS DETALLADO DE LA PREDICCIÓN:")
        print(f"   Raw prediction: {details['raw_prediction']} pts")
        print(f"   H2H adjusted: {details['h2h_adjusted_prediction']:.1f} pts")
        print(f"   Final prediction: {details['final_prediction']:.1f} pts")
        print(f"   Historical average: {details['actual_stats_mean']:.1f} pts")
        print(f"   Confidence: {prediction_result['confidence_percentage']:.1f}%")
        
        print(f"\n📊 COMPONENTES DE LA PREDICCIÓN:")
        print(f"   Raw → H2H: {details['raw_prediction']} → {details['h2h_adjusted_prediction']:.1f} (factor: {details['h2h_stats']['h2h_factor']:.3f})")
        print(f"   H2H → Final: {details['h2h_adjusted_prediction']:.1f} → {details['final_prediction']:.1f}")
        
        print(f"\n🎯 FACTORES DE CONFIANZA:")
        print(f"   Tolerance applied: {details['tolerance_applied']}")
        print(f"   Historical games used: {details['historical_games_used']}")
        print(f"   Data quality: {details['performance_metrics']['confidence_factors']['data_quality']}")
        
        print(f"\n📈 TENDENCIAS RECIENTES:")
        print(f"   Últimos 5 juegos: {details['last_5_games']['mean']:.1f} ± {details['last_5_games']['std']:.1f}")
        print(f"   Últimos 10 juegos: {details['last_10_games']['mean']:.1f} ± {details['last_10_games']['std']:.1f}")
        print(f"   Trend 5 games: {details['trend_analysis']['trend_5_games']:+.1f}")
        print(f"   Recent form: {details['trend_analysis']['recent_form']:.1f}")
        print(f"   Consistency score: {details['trend_analysis']['consistency_score']:.1f}%")
        
        print(f"\n🔄 H2H STATS DETALLADAS:")
        h2h = details['h2h_stats']
        print(f"   Games found: {h2h['games_found']}")
        print(f"   H2H mean: {h2h['h2h_mean']:.1f} pts")
        print(f"   H2H std: {h2h['h2h_std']:.1f} pts")
        print(f"   H2H factor: {h2h['h2h_factor']:.3f}")
        print(f"   Consistency score: {h2h['consistency_score']:.1f}%")
        print(f"   Last 5 H2H: {h2h['last_5_mean']:.1f} pts")
        print(f"   Last 10 H2H: {h2h['last_10_mean']:.1f} pts")
        
        # 5. ANÁLISIS DE CONSISTENCIA
        print(f"\n⚖️ 5. ANÁLISIS DE CONSISTENCIA:")
        print("-" * 40)
        
        historical_avg = details['actual_stats_mean']
        final_pred = details['final_prediction']
        h2h_avg = h2h['h2h_mean']
        
        print(f"   Promedio histórico: {historical_avg:.1f} pts")
        print(f"   Promedio H2H vs MIN: {h2h_avg:.1f} pts")
        print(f"   Predicción final: {final_pred:.1f} pts")
        
        # Verificar consistencia
        pred_vs_historical = final_pred - historical_avg
        pred_vs_h2h = final_pred - h2h_avg
        
        print(f"\n   Diferencias:")
        print(f"   Predicción vs Histórico: {pred_vs_historical:+.1f} pts")
        print(f"   Predicción vs H2H: {pred_vs_h2h:+.1f} pts")
        
        # Evaluar si la predicción es razonable
        if abs(pred_vs_historical) <= 5:
            print(f"   ✅ Predicción razonable vs histórico")
        else:
            print(f"   ⚠️ Predicción alta vs histórico")
            
        if abs(pred_vs_h2h) <= 3:
            print(f"   ✅ Predicción consistente con H2H")
        else:
            print(f"   ⚠️ Predicción diferente al H2H")
        
        # 6. RESULTADO FINAL
        print(f"\n📋 6. RESULTADO FINAL:")
        print("-" * 40)
        print(f"✅ Predicción: {final_pred:.1f} pts")
        print(f"✅ Confianza: {prediction_result['confidence_percentage']:.1f}%")
        print(f"✅ Factor H2H: {h2h['h2h_factor']:.3f}")
        print(f"✅ Consistencia: {h2h['consistency_score']:.1f}%")
        
        # Guardar resultado completo
        with open('shai_vs_minnesota_debug.json', 'w') as f:
            json.dump(prediction_result, f, indent=2, default=str)
        print(f"\n💾 Resultado completo guardado en: shai_vs_minnesota_debug.json")
        
    else:
        print("❌ Error en la predicción")

if __name__ == "__main__":
    debug_shai_vs_minnesota()
