#!/usr/bin/env python3
"""
Prueba específica: Jokić vs Orlando Magic
========================================

Prueba específica para verificar las predicciones de Jokić contra Orlando Magic
y comparar con Shai contra Orlando Magic.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import logging

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Importar modelos y data loaders
from app.architectures.basketball.pipelines.predict.players.pts_predictor import PTSPredictor

def test_jokic_vs_orlando():
    """Prueba específica Jokić vs Orlando Magic"""
    
    print("🎯 PRUEBA ESPECÍFICA: JOKIĆ vs ORLANDO MAGIC")
    print("=" * 50)
    
    # Crear predictor
    predictor = PTSPredictor()
    
    # Mockup específico: Denver vs Orlando Magic
    mock_game = {
        "gameId": "sr:match:12345",
        "scheduled": "2024-01-15T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Denver Nuggets",
            "alias": "DEN",
            "players": [
                {
                    "playerId": "sr:player:jokic",
                    "fullName": "Nikola Jokic",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "15",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Orlando Magic", 
            "alias": "ORL",
            "players": [
                {
                    "playerId": "sr:player:banchero",
                    "fullName": "Paolo Banchero",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "5",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "Ball Arena",
            "capacity": 19520
        }
    }
    
    print("\n🏀 PRUEBA 1: JOKIĆ vs ORLANDO MAGIC")
    print("-" * 40)
    
    # Probar Jokić vs Orlando
    jokic_result = predictor.predict_game(mock_game, "Nikola Jokic")
    
    if jokic_result is not None:
        print("✅ Predicción exitosa para Jokić:")
        print(f"   Raw prediction: {jokic_result['prediction_details']['raw_prediction']} pts")
        print(f"   H2H adjusted: {jokic_result['prediction_details']['h2h_adjusted_prediction']:.1f} pts")
        print(f"   Final prediction: {jokic_result['prediction_details']['final_prediction']:.1f} pts")
        print(f"   Historical average: {jokic_result['prediction_details']['actual_stats_mean']:.1f} pts")
        print(f"   H2H factor: {jokic_result['prediction_details']['h2h_stats']['h2h_factor']:.3f}")
        print(f"   H2H games found: {jokic_result['prediction_details']['h2h_stats']['games_found']}")
        print(f"   H2H mean: {jokic_result['prediction_details']['h2h_stats']['h2h_mean']:.1f} pts")
    else:
        print("❌ Error en predicción de Jokić")
    
    print("\n🏀 PRUEBA 2: SHAI vs ORLANDO MAGIC")
    print("-" * 40)
    
    # Cambiar el mockup para Shai vs Orlando
    mock_game_shai = {
        "gameId": "sr:match:12346",
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
            "name": "Orlando Magic", 
            "alias": "ORL",
            "players": [
                {
                    "playerId": "sr:player:banchero",
                    "fullName": "Paolo Banchero",
                    "position": "F",
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
    
    # Probar Shai vs Orlando
    shai_result = predictor.predict_game(mock_game_shai, "Shai Gilgeous-Alexander")
    
    if shai_result is not None:
        print("✅ Predicción exitosa para Shai:")
        print(f"   Raw prediction: {shai_result['prediction_details']['raw_prediction']} pts")
        print(f"   H2H adjusted: {shai_result['prediction_details']['h2h_adjusted_prediction']:.1f} pts")
        print(f"   Final prediction: {shai_result['prediction_details']['final_prediction']:.1f} pts")
        print(f"   Historical average: {shai_result['prediction_details']['actual_stats_mean']:.1f} pts")
        print(f"   H2H factor: {shai_result['prediction_details']['h2h_stats']['h2h_factor']:.3f}")
        print(f"   H2H games found: {shai_result['prediction_details']['h2h_stats']['games_found']}")
        print(f"   H2H mean: {shai_result['prediction_details']['h2h_stats']['h2h_mean']:.1f} pts")
    else:
        print("❌ Error en predicción de Shai")
    
    # Comparación
    if jokic_result is not None and shai_result is not None:
        print("\n📊 COMPARACIÓN:")
        print("-" * 20)
        print(f"Jokic vs ORL: {jokic_result['prediction_details']['final_prediction']:.1f} pts")
        print(f"Shai vs ORL:  {shai_result['prediction_details']['final_prediction']:.1f} pts")
        print(f"Diferencia:   {jokic_result['prediction_details']['final_prediction'] - shai_result['prediction_details']['final_prediction']:+.1f} pts")
        
        print(f"\nH2H Factors:")
        print(f"Jokic H2H factor: {jokic_result['prediction_details']['h2h_stats']['h2h_factor']:.3f}")
        print(f"Shai H2H factor:  {shai_result['prediction_details']['h2h_stats']['h2h_factor']:.3f}")
        
        print(f"\nHistorical Averages:")
        print(f"Jokic historical: {jokic_result['prediction_details']['actual_stats_mean']:.1f} pts")
        print(f"Shai historical:  {shai_result['prediction_details']['actual_stats_mean']:.1f} pts")
        
        # Verificar consistencia
        jokic_higher_pred = jokic_result['prediction_details']['final_prediction'] > shai_result['prediction_details']['final_prediction']
        shai_higher_real = shai_result['prediction_details']['actual_stats_mean'] > jokic_result['prediction_details']['actual_stats_mean']
        
        print(f"\n🔍 VERIFICACIÓN:")
        print(f"Jokic predice más alto: {jokic_higher_pred}")
        print(f"Shai anota más: {shai_higher_real}")
        
        if jokic_higher_pred and shai_higher_real:
            print("❌ PROBLEMA: Inconsistencia detectada")
        else:
            print("✅ CONSISTENTE: Predicciones alineadas con la realidad")

if __name__ == "__main__":
    test_jokic_vs_orlando()
