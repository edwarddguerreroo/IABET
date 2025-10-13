#!/usr/bin/env python3
"""
Comparación: Modelo CON vs SIN correcciones
==========================================

Compara el comportamiento del modelo con y sin las correcciones aplicadas.
"""

import pandas as pd
import numpy as np

def compare_models():
    """Compara modelos con y sin correcciones"""
    
    print("🔄 COMPARACIÓN: MODELO CON vs SIN CORRECCIONES")
    print("=" * 60)
    
    # Cargar predicciones actuales (sin correcciones)
    current_df = pd.read_csv('app/architectures/basketball/results/pts_model/predictions.csv')
    
    print("📊 ESTADÍSTICAS GENERALES:")
    print(f"{'Métrica':<25} | {'Sin Correcciones':<15} | {'Diferencia':<10}")
    print("-" * 60)
    
    # Estadísticas básicas
    mae_current = current_df['abs_error'].mean()
    bias_current = current_df['error'].mean()
    pred_mean_current = current_df['points_predicted'].mean()
    pred_std_current = current_df['points_predicted'].std()
    
    print(f"{'MAE':<25} | {mae_current:<15.3f} | {'Baseline':<10}")
    print(f"{'Bias General':<25} | {bias_current:<15.3f} | {'Baseline':<10}")
    print(f"{'Pred Media':<25} | {pred_mean_current:<15.3f} | {'Baseline':<10}")
    print(f"{'Pred Std':<25} | {pred_std_current:<15.3f} | {'Baseline':<10}")
    
    # Análisis por rangos
    print("\n🎯 ANÁLISIS POR RANGOS:")
    print(f"{'Rango':<20} | {'MAE':<8} | {'Bias':<8} | {'Count':<8} | {'%':<6}")
    print("-" * 60)
    
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
    
    for min_pts, max_pts, label in ranges:
        mask = (current_df['points'] >= min_pts) & (current_df['points'] < max_pts)
        subset = current_df[mask]
        
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            count = len(subset)
            percentage = (count / len(current_df)) * 100
            
            print(f"{label:<20} | {mae:<8.2f} | {bias:<8.2f} | {count:<8,} | {percentage:<6.1f}")
    
    # Análisis de jugadores específicos
    print("\n⭐ JUGADORES ESPECÍFICOS:")
    
    # Top scorers
    top_scorers = current_df.groupby('player').agg({
        'points': ['mean', 'count'],
        'points_predicted': 'mean',
        'abs_error': 'mean',
        'error': 'mean'
    }).round(2)
    
    top_scorers.columns = ['Real_Avg', 'Games', 'Pred_Avg', 'MAE', 'Bias']
    top_scorers = top_scorers[top_scorers['Games'] >= 20].sort_values('Real_Avg', ascending=False)
    
    print(f"\n{'Jugador':<25} | {'Real':<6} | {'Pred':<6} | {'MAE':<6} | {'Bias':<8}")
    print("-" * 65)
    
    for player, row in top_scorers.head(10).iterrows():
        print(f"{player:<25} | {row['Real_Avg']:<6.1f} | {row['Pred_Avg']:<6.1f} | {row['MAE']:<6.2f} | {row['Bias']:<8.2f}")
    
    # Análisis específico de Jokić y Shai
    print("\n🎯 ANÁLISIS ESPECÍFICO:")
    
    jokic_data = current_df[current_df['player'] == 'Nikola Jokić']
    shai_data = current_df[current_df['player'] == 'Shai Gilgeous-Alexander']
    
    if len(jokic_data) > 0:
        print(f"\nNikola Jokić:")
        print(f"  Juegos: {len(jokic_data)}")
        print(f"  Real promedio: {jokic_data['points'].mean():.2f}")
        print(f"  Pred promedio: {jokic_data['points_predicted'].mean():.2f}")
        print(f"  MAE: {jokic_data['abs_error'].mean():.2f}")
        print(f"  Bias: {jokic_data['error'].mean():.2f}")
    
    if len(shai_data) > 0:
        print(f"\nShai Gilgeous-Alexander:")
        print(f"  Juegos: {len(shai_data)}")
        print(f"  Real promedio: {shai_data['points'].mean():.2f}")
        print(f"  Pred promedio: {shai_data['points_predicted'].mean():.2f}")
        print(f"  MAE: {shai_data['abs_error'].mean():.2f}")
        print(f"  Bias: {shai_data['error'].mean():.2f}")
    
    # Verificación del problema original
    print("\n🔍 VERIFICACIÓN DEL PROBLEMA ORIGINAL:")
    if len(jokic_data) > 0 and len(shai_data) > 0:
        jokic_pred_avg = jokic_data['points_predicted'].mean()
        shai_pred_avg = shai_data['points_predicted'].mean()
        jokic_real_avg = jokic_data['points'].mean()
        shai_real_avg = shai_data['points'].mean()
        
        print(f"Jokic predice más alto que Shai: {jokic_pred_avg:.1f} > {shai_pred_avg:.1f} = {jokic_pred_avg > shai_pred_avg}")
        print(f"Shai anota más que Jokic: {shai_real_avg:.1f} > {jokic_real_avg:.1f} = {shai_real_avg > jokic_real_avg}")
        
        if jokic_pred_avg > shai_pred_avg and shai_real_avg > jokic_real_avg:
            print("❌ PROBLEMA PERSISTE: Inconsistencia en las predicciones")
        else:
            print("✅ PROBLEMA RESUELTO: Predicciones consistentes")
    
    # Análisis de correlación
    correlation = current_df['points'].corr(current_df['points_predicted'])
    print(f"\n📊 Correlación real vs predicho: {correlation:.4f}")
    
    # Análisis de outliers
    high_error_mask = current_df['abs_error'] > 10
    high_error_count = high_error_mask.sum()
    high_error_pct = (high_error_count / len(current_df)) * 100
    
    print(f"\n🚨 Outliers (error > 10): {high_error_count:,} ({high_error_pct:.2f}%)")
    
    # Resumen de hallazgos
    print("\n📋 RESUMEN DE HALLAZGOS:")
    print("=" * 40)
    
    # Sesgo por rangos
    low_range_mask = current_df['points'] < 10
    high_range_mask = current_df['points'] >= 25
    
    low_bias = current_df[low_range_mask]['error'].mean() if low_range_mask.sum() > 0 else 0
    high_bias = current_df[high_range_mask]['error'].mean() if high_range_mask.sum() > 0 else 0
    
    print(f"1. Sesgo en rangos bajos (<10 pts): {low_bias:+.2f}")
    print(f"2. Sesgo en rangos altos (≥25 pts): {high_bias:+.2f}")
    print(f"3. MAE general: {mae_current:.3f}")
    print(f"4. Correlación: {correlation:.4f}")
    
    if len(jokic_data) > 0 and len(shai_data) > 0:
        print(f"5. Problema Jokić vs Shai: {'PERSISTE' if jokic_data['points_predicted'].mean() > shai_data['points_predicted'].mean() and shai_data['points'].mean() > jokic_data['points'].mean() else 'RESUELTO'}")

if __name__ == "__main__":
    compare_models()
