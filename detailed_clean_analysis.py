#!/usr/bin/env python3
"""
Análisis Detallado del Modelo Sin Correcciones
==============================================

Análisis profundo del comportamiento del modelo base para identificar
patrones y problemas específicos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_analysis():
    """Análisis detallado del modelo sin correcciones"""
    
    print("🔬 ANÁLISIS DETALLADO DEL MODELO SIN CORRECCIONES")
    print("=" * 60)
    
    # Cargar datos
    df = pd.read_csv('app/architectures/basketball/results/pts_model/predictions.csv')
    
    # Análisis de distribución de errores
    print("\n📊 DISTRIBUCIÓN DE ERRORES:")
    print(f"Error mínimo: {df['error'].min():.2f}")
    print(f"Error máximo: {df['error'].max():.2f}")
    print(f"Error mediano: {df['error'].median():.2f}")
    print(f"Error std: {df['error'].std():.2f}")
    
    # Percentiles de error
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles de error:")
    for p in percentiles:
        val = np.percentile(df['error'], p)
        print(f"  P{p}: {val:+.2f}")
    
    # Análisis de sesgo por rangos más detallado
    print("\n🎯 SESGO DETALLADO POR RANGOS:")
    
    ranges = [
        (0, 2, "Muy bajo (0-1)"),
        (2, 5, "Bajo (2-4)"),
        (5, 10, "Medio-bajo (5-9)"),
        (10, 15, "Medio (10-14)"),
        (15, 20, "Medio-alto (15-19)"),
        (20, 25, "Alto (20-24)"),
        (25, 30, "Muy alto (25-29)"),
        (30, 40, "Élite (30-39)"),
        (40, 100, "Histórico (40+)")
    ]
    
    print(f"{'Rango':<20} | {'Count':<8} | {'Real':<6} | {'Pred':<6} | {'MAE':<6} | {'Bias':<8} | {'%':<6}")
    print("-" * 80)
    
    for min_pts, max_pts, label in ranges:
        mask = (df['points'] >= min_pts) & (df['points'] < max_pts)
        subset = df[mask]
        
        if len(subset) > 0:
            real_mean = subset['points'].mean()
            pred_mean = subset['points_predicted'].mean()
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            count = len(subset)
            percentage = (count / len(df)) * 100
            
            print(f"{label:<20} | {count:<8,} | {real_mean:<6.1f} | {pred_mean:<6.1f} | {mae:<6.2f} | {bias:<8.2f} | {percentage:<6.1f}")
    
    # Análisis de jugadores específicos más detallado
    print("\n⭐ ANÁLISIS DETALLADO DE JUGADORES:")
    
    # Top 20 scorers
    top_scorers = df.groupby('player').agg({
        'points': ['mean', 'count', 'std'],
        'points_predicted': ['mean', 'std'],
        'abs_error': 'mean',
        'error': ['mean', 'std']
    }).round(2)
    
    top_scorers.columns = ['Real_Avg', 'Games', 'Real_Std', 'Pred_Avg', 'Pred_Std', 'MAE', 'Bias', 'Bias_Std']
    top_scorers = top_scorers[top_scorers['Games'] >= 20].sort_values('Real_Avg', ascending=False)
    
    print(f"\n{'Jugador':<25} | {'Real':<6} | {'Pred':<6} | {'MAE':<6} | {'Bias':<8} | {'Games':<6}")
    print("-" * 75)
    
    for player, row in top_scorers.head(20).iterrows():
        print(f"{player:<25} | {row['Real_Avg']:<6.1f} | {row['Pred_Avg']:<6.1f} | {row['MAE']:<6.2f} | {row['Bias']:<8.2f} | {row['Games']:<6.0f}")
    
    # Análisis específico de Jokić
    print("\n🎯 ANÁLISIS ESPECÍFICO DE NIKOLA JOKIĆ:")
    jokic_data = df[df['player'] == 'Nikola Jokić']
    
    if len(jokic_data) > 0:
        print(f"Total de juegos: {len(jokic_data)}")
        print(f"Puntos reales - Media: {jokic_data['points'].mean():.2f}, Mediana: {jokic_data['points'].median():.2f}")
        print(f"Puntos predichos - Media: {jokic_data['points_predicted'].mean():.2f}, Mediana: {jokic_data['points_predicted'].median():.2f}")
        print(f"MAE: {jokic_data['abs_error'].mean():.2f}")
        print(f"Bias: {jokic_data['error'].mean():.2f}")
        
        # Análisis por rangos para Jokić
        print(f"\nJokić por rangos:")
        for min_pts, max_pts, label in ranges:
            mask = (jokic_data['points'] >= min_pts) & (jokic_data['points'] < max_pts)
            subset = jokic_data[mask]
            if len(subset) > 0:
                print(f"  {label:<20} | Count: {len(subset):3} | MAE: {subset['abs_error'].mean():5.2f} | Bias: {subset['error'].mean():+6.2f}")
        
        # Casos problemáticos de Jokić
        print(f"\nCasos problemáticos de Jokić (error > 5):")
        high_error_jokic = jokic_data[jokic_data['abs_error'] > 5].sort_values('abs_error', ascending=False)
        for _, row in high_error_jokic.head(10).iterrows():
            print(f"  Real: {row['points']:3.0f} | Pred: {row['points_predicted']:5.1f} | Error: {row['abs_error']:5.1f}")
    
    # Análisis de Shai
    print("\n🎯 ANÁLISIS ESPECÍFICO DE SHAI GILGEOUS-ALEXANDER:")
    shai_data = df[df['player'] == 'Shai Gilgeous-Alexander']
    
    if len(shai_data) > 0:
        print(f"Total de juegos: {len(shai_data)}")
        print(f"Puntos reales - Media: {shai_data['points'].mean():.2f}, Mediana: {shai_data['points'].median():.2f}")
        print(f"Puntos predichos - Media: {shai_data['points_predicted'].mean():.2f}, Mediana: {shai_data['points_predicted'].median():.2f}")
        print(f"MAE: {shai_data['abs_error'].mean():.2f}")
        print(f"Bias: {shai_data['error'].mean():.2f}")
        
        # Casos problemáticos de Shai
        print(f"\nCasos problemáticos de Shai (error > 5):")
        high_error_shai = shai_data[shai_data['abs_error'] > 5].sort_values('abs_error', ascending=False)
        for _, row in high_error_shai.head(10).iterrows():
            print(f"  Real: {row['points']:3.0f} | Pred: {row['points_predicted']:5.1f} | Error: {row['abs_error']:5.1f}")
    
    # Comparación directa Jokić vs Shai
    print("\n⚖️ COMPARACIÓN DIRECTA JOKIĆ vs SHAI:")
    if len(jokic_data) > 0 and len(shai_data) > 0:
        print(f"{'Métrica':<20} | {'Jokić':<10} | {'Shai':<10} | {'Diferencia':<10}")
        print("-" * 60)
        
        jokic_real = jokic_data['points'].mean()
        shai_real = shai_data['points'].mean()
        jokic_pred = jokic_data['points_predicted'].mean()
        shai_pred = shai_data['points_predicted'].mean()
        jokic_mae = jokic_data['abs_error'].mean()
        shai_mae = shai_data['abs_error'].mean()
        jokic_bias = jokic_data['error'].mean()
        shai_bias = shai_data['error'].mean()
        
        print(f"{'Real promedio':<20} | {jokic_real:<10.2f} | {shai_real:<10.2f} | {jokic_real-shai_real:<10.2f}")
        print(f"{'Pred promedio':<20} | {jokic_pred:<10.2f} | {shai_pred:<10.2f} | {jokic_pred-shai_pred:<10.2f}")
        print(f"{'MAE':<20} | {jokic_mae:<10.2f} | {shai_mae:<10.2f} | {jokic_mae-shai_mae:<10.2f}")
        print(f"{'Bias':<20} | {jokic_bias:<10.2f} | {shai_bias:<10.2f} | {jokic_bias-shai_bias:<10.2f}")
        
        # Verificar el problema
        print(f"\n🔍 VERIFICACIÓN DEL PROBLEMA:")
        print(f"Jokic predice más alto que Shai: {jokic_pred:.1f} > {shai_pred:.1f} = {jokic_pred > shai_pred}")
        print(f"Shai anota más que Jokic: {shai_real:.1f} > {jokic_real:.1f} = {shai_real > jokic_real}")
        
        if jokic_pred > shai_pred and shai_real > jokic_real:
            print("❌ PROBLEMA CONFIRMADO: Inconsistencia en las predicciones")
            print(f"   Diferencia en predicciones: {jokic_pred - shai_pred:.1f} puntos")
            print(f"   Diferencia en realidad: {shai_real - jokic_real:.1f} puntos")
        else:
            print("✅ NO HAY PROBLEMA: Predicciones consistentes")
    
    # Análisis de correlación por rangos
    print("\n📊 CORRELACIÓN POR RANGOS:")
    for min_pts, max_pts, label in ranges:
        mask = (df['points'] >= min_pts) & (df['points'] < max_pts)
        subset = df[mask]
        
        if len(subset) > 10:  # Solo si hay suficientes datos
            corr = subset['points'].corr(subset['points_predicted'])
            print(f"{label:<20} | Correlación: {corr:.4f} | Count: {len(subset):,}")
    
    # Análisis de outliers
    print("\n🚨 ANÁLISIS DE OUTLIERS:")
    
    # Outliers por error absoluto
    error_thresholds = [5, 10, 15, 20]
    for threshold in error_thresholds:
        outliers = df[df['abs_error'] > threshold]
        count = len(outliers)
        percentage = (count / len(df)) * 100
        print(f"Error > {threshold:2d}: {count:4,} ({percentage:5.2f}%)")
    
    # Top outliers
    print(f"\nTop 15 peores predicciones:")
    worst_predictions = df.nlargest(15, 'abs_error')
    for _, row in worst_predictions.iterrows():
        print(f"{row['player']:<25} | Real: {row['points']:3.0f} | Pred: {row['points_predicted']:5.1f} | Error: {row['abs_error']:5.1f}")

if __name__ == "__main__":
    detailed_analysis()
