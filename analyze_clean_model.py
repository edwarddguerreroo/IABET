#!/usr/bin/env python3
"""
Análisis del modelo de puntos SIN correcciones
==============================================

Analiza el comportamiento del modelo base sin pesos adaptativos,
correcciones por rango, ni penalizaciones especiales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_clean_model():
    """Analiza el modelo sin correcciones"""
    
    print("🔍 ANÁLISIS DEL MODELO SIN CORRECCIONES")
    print("=" * 50)
    
    # Cargar predicciones
    predictions_df = pd.read_csv('app/architectures/basketball/results/pts_model/predictions.csv')
    
    print(f"📊 Total de predicciones: {len(predictions_df):,}")
    print(f"📅 Rango de fechas: {predictions_df['Date'].min()} a {predictions_df['Date'].max()}")
    print(f"👥 Jugadores únicos: {predictions_df['player'].nunique():,}")
    
    # Estadísticas básicas
    print("\n📈 ESTADÍSTICAS BÁSICAS:")
    print(f"Puntos reales - Media: {predictions_df['points'].mean():.2f}, Mediana: {predictions_df['points'].median():.2f}")
    print(f"Puntos predichos - Media: {predictions_df['points_predicted'].mean():.2f}, Mediana: {predictions_df['points_predicted'].median():.2f}")
    print(f"MAE: {predictions_df['abs_error'].mean():.3f}")
    print(f"Error medio: {predictions_df['error'].mean():.3f}")
    
    # Análisis por rangos
    print("\n🎯 ANÁLISIS POR RANGOS DE PUNTOS:")
    
    # Definir rangos
    ranges = [
        (0, 5, "Suplentes (0-4)"),
        (5, 10, "Rotación (5-9)"),
        (10, 15, "Importantes (10-14)"),
        (15, 20, "Clave (15-19)"),
        (20, 25, "Estrellas (20-24)"),
        (25, 30, "Superestrellas (25-29)"),
        (30, 40, "Élite (30-39)"),
        (40, 100, "Histórico (40+)")
    ]
    
    range_analysis = []
    
    for min_pts, max_pts, label in ranges:
        mask = (predictions_df['points'] >= min_pts) & (predictions_df['points'] < max_pts)
        subset = predictions_df[mask]
        
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            count = len(subset)
            percentage = (count / len(predictions_df)) * 100
            
            range_analysis.append({
                'Rango': label,
                'Min': min_pts,
                'Max': max_pts,
                'Count': count,
                'Percentage': percentage,
                'MAE': mae,
                'Bias': bias,
                'Real_Mean': subset['points'].mean(),
                'Pred_Mean': subset['points_predicted'].mean()
            })
            
            print(f"{label:20} | Count: {count:6,} ({percentage:5.1f}%) | MAE: {mae:5.2f} | Bias: {bias:+6.2f}")
    
    range_df = pd.DataFrame(range_analysis)
    
    # Análisis de jugadores específicos
    print("\n⭐ ANÁLISIS DE JUGADORES ESPECÍFICOS:")
    
    # Top scorers
    top_scorers = predictions_df.groupby('player').agg({
        'points': ['mean', 'count'],
        'points_predicted': 'mean',
        'abs_error': 'mean',
        'error': 'mean'
    }).round(2)
    
    top_scorers.columns = ['Real_Avg', 'Games', 'Pred_Avg', 'MAE', 'Bias']
    top_scorers = top_scorers[top_scorers['Games'] >= 20].sort_values('Real_Avg', ascending=False)
    
    print("\n🏆 TOP 10 SCORERS (20+ juegos):")
    for i, (player, row) in enumerate(top_scorers.head(10).iterrows()):
        print(f"{i+1:2d}. {player:25} | Real: {row['Real_Avg']:5.1f} | Pred: {row['Pred_Avg']:5.1f} | MAE: {row['MAE']:5.2f} | Bias: {row['Bias']:+6.2f}")
    
    # Análisis de Jokić específicamente
    print("\n🎯 ANÁLISIS ESPECÍFICO DE NIKOLA JOKIĆ:")
    jokic_data = predictions_df[predictions_df['player'] == 'Nikola Jokić']
    
    if len(jokic_data) > 0:
        print(f"Juegos analizados: {len(jokic_data)}")
        print(f"Puntos reales promedio: {jokic_data['points'].mean():.2f}")
        print(f"Puntos predichos promedio: {jokic_data['points_predicted'].mean():.2f}")
        print(f"MAE: {jokic_data['abs_error'].mean():.2f}")
        print(f"Bias: {jokic_data['error'].mean():.2f}")
        
        # Análisis por rangos para Jokić
        print("\nJokić por rangos:")
        for min_pts, max_pts, label in ranges:
            mask = (jokic_data['points'] >= min_pts) & (jokic_data['points'] < max_pts)
            subset = jokic_data[mask]
            if len(subset) > 0:
                print(f"{label:20} | Count: {len(subset):3} | MAE: {subset['abs_error'].mean():5.2f} | Bias: {subset['error'].mean():+6.2f}")
    
    # Análisis de Shai Gilgeous-Alexander
    print("\n🎯 ANÁLISIS ESPECÍFICO DE SHAI GILGEOUS-ALEXANDER:")
    shai_data = predictions_df[predictions_df['player'] == 'Shai Gilgeous-Alexander']
    
    if len(shai_data) > 0:
        print(f"Juegos analizados: {len(shai_data)}")
        print(f"Puntos reales promedio: {shai_data['points'].mean():.2f}")
        print(f"Puntos predichos promedio: {shai_data['points_predicted'].mean():.2f}")
        print(f"MAE: {shai_data['abs_error'].mean():.2f}")
        print(f"Bias: {shai_data['error'].mean():.2f}")
    
    # Comparación Jokić vs Shai
    print("\n⚖️ COMPARACIÓN JOKIĆ vs SHAI:")
    if len(jokic_data) > 0 and len(shai_data) > 0:
        print(f"Jokic  - Real: {jokic_data['points'].mean():5.1f} | Pred: {jokic_data['points_predicted'].mean():5.1f} | Bias: {jokic_data['error'].mean():+6.2f}")
        print(f"Shai   - Real: {shai_data['points'].mean():5.1f} | Pred: {shai_data['points_predicted'].mean():5.1f} | Bias: {shai_data['error'].mean():+6.2f}")
        
        # Verificar si el problema persiste
        jokic_pred_avg = jokic_data['points_predicted'].mean()
        shai_pred_avg = shai_data['points_predicted'].mean()
        jokic_real_avg = jokic_data['points'].mean()
        shai_real_avg = shai_data['points'].mean()
        
        print(f"\n🔍 VERIFICACIÓN DEL PROBLEMA:")
        print(f"Jokic predice más alto que Shai: {jokic_pred_avg > shai_pred_avg}")
        print(f"Pero Shai anota más que Jokic: {shai_real_avg > jokic_real_avg}")
        print(f"Inconsistencia: {jokic_pred_avg > shai_pred_avg and shai_real_avg > jokic_real_avg}")
    
    # Análisis de correlación
    print("\n📊 CORRELACIÓN:")
    correlation = predictions_df['points'].corr(predictions_df['points_predicted'])
    print(f"Correlación real vs predicho: {correlation:.4f}")
    
    # Análisis de outliers
    print("\n🚨 ANÁLISIS DE OUTLIERS:")
    high_error_mask = predictions_df['abs_error'] > 10
    high_error_data = predictions_df[high_error_mask]
    
    print(f"Predicciones con error > 10 puntos: {len(high_error_data)} ({len(high_error_data)/len(predictions_df)*100:.2f}%)")
    
    if len(high_error_data) > 0:
        print("\nTop 10 peores predicciones:")
        worst_predictions = high_error_data.nlargest(10, 'abs_error')
        for _, row in worst_predictions.iterrows():
            print(f"{row['player']:25} | Real: {row['points']:3.0f} | Pred: {row['points_predicted']:5.1f} | Error: {row['abs_error']:5.1f}")
    
    return range_df, top_scorers

if __name__ == "__main__":
    range_df, top_scorers = analyze_clean_model()
