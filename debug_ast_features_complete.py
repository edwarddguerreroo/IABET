"""
🔍 DEBUG COMPLETO - ASSISTS FEATURE ENGINEER
═══════════════════════════════════════════════════════════════════════════════

Script de debugging exhaustivo para el feature engineer de AST.
Verifica:
- Carga de datos y índices
- Generación de features por grupo
- Porcentaje de NaN en cada feature
- Orden de ejecución
- Integridad de datos
- Propagación correcta de NaN (sin fallbacks)

Autor: Sistema de Debug Automatizado
Fecha: 2025-01-24
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.ast.features_ast import AssistsFeatureEngineer

def print_section(title: str, char: str = "═"):
    """Imprime una sección con formato"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")

def print_subsection(title: str):
    """Imprime una subsección con formato"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")

def analyze_feature_quality(df: pd.DataFrame, feature_name: str) -> dict:
    """Analiza la calidad de una feature"""
    if feature_name not in df.columns:
        return {
            'exists': False,
            'total_count': 0,
            'valid_count': 0,
            'nan_count': 0,
            'nan_percentage': 100.0,
            'zero_count': 0,
            'zero_percentage': 0.0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    series = df[feature_name]
    total = len(series)
    valid = series.notna().sum()
    nan_count = series.isna().sum()
    zero_count = (series == 0).sum()
    
    return {
        'exists': True,
        'total_count': total,
        'valid_count': valid,
        'nan_count': nan_count,
        'nan_percentage': (nan_count / total * 100) if total > 0 else 0,
        'zero_count': zero_count,
        'zero_percentage': (zero_count / total * 100) if total > 0 else 0,
        'mean': series.mean() if valid > 0 else np.nan,
        'std': series.std() if valid > 0 else np.nan,
        'min': series.min() if valid > 0 else np.nan,
        'max': series.max() if valid > 0 else np.nan
    }

def print_feature_stats(stats: dict, feature_name: str):
    """Imprime estadísticas de una feature"""
    if not stats['exists']:
        print(f"❌ {feature_name}: NO EXISTE")
        return
    
    # Determinar el emoji según el porcentaje de NaN
    if stats['nan_percentage'] < 10:
        emoji = "✅"
    elif stats['nan_percentage'] < 30:
        emoji = "⚠️"
    else:
        emoji = "🔴"
    
    print(f"{emoji} {feature_name}:")
    print(f"   Valid: {stats['valid_count']:,}/{stats['total_count']:,} ({100-stats['nan_percentage']:.1f}%)")
    print(f"   NaN: {stats['nan_count']:,} ({stats['nan_percentage']:.1f}%)")
    if stats['zero_count'] > 0:
        print(f"   Zeros: {stats['zero_count']:,} ({stats['zero_percentage']:.1f}%)")
    if pd.notna(stats['mean']):
        print(f"   Stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")

def main():
    print_section("🚀 INICIO DEL DEBUG - ASSISTS FEATURE ENGINEER", "═")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. CARGA DE DATOS
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("1️⃣ CARGA DE DATOS", "═")
    
    try:
        print("📂 Cargando datos desde NBADataLoader...")
        data_loader = NBADataLoader()
        
        # Cargar datos principales (retorna tupla)
        result = data_loader.load_data(use_quarters=False, force_reload=False)
        if isinstance(result, tuple):
            df_players, df_teams, _, _ = result
        else:
            df_players = result
            df_teams = None
        
        print(f"✅ Datos de jugadores cargados: {len(df_players):,} filas")
        if df_teams is not None:
            print(f"✅ Datos de equipos cargados: {len(df_teams):,} filas")
        else:
            print(f"⚠️  Datos de equipos no disponibles")
        
        # Cargar datos de quarters
        result_quarters = data_loader.load_data(use_quarters=True, force_reload=False)
        if isinstance(result_quarters, tuple):
            df_quarters, _, _, _ = result_quarters
        else:
            df_quarters = result_quarters
        
        print(f"✅ Datos de quarters cargados: {len(df_quarters):,} filas")
        
        # Información básica
        print_subsection("📊 INFORMACIÓN BÁSICA DEL DATASET")
        print(f"Shape: {df_players.shape}")
        print(f"Columnas: {len(df_players.columns)}")
        print(f"Jugadores únicos: {df_players['player'].nunique()}")
        print(f"Temporadas: {df_players['Season'].nunique() if 'Season' in df_players.columns else 'N/A'}")
        
        # Verificar columna de asistencias
        if 'assists' in df_players.columns:
            print(f"\n✅ Columna 'assists' encontrada")
            print(f"   Total asistencias: {df_players['assists'].sum():,.0f}")
            print(f"   Promedio por juego: {df_players['assists'].mean():.2f}")
            print(f"   Máximo en un juego: {df_players['assists'].max():.0f}")
        else:
            print("❌ ERROR: Columna 'assists' no encontrada!")
            return
        
        # Verificar índices
        print_subsection("🔍 VERIFICACIÓN DE ÍNDICES")
        print(f"Tipo de índice: {type(df_players.index)}")
        print(f"Nombre del índice: {df_players.index.name}")
        print(f"Índices únicos: {df_players.index.is_unique}")
        print(f"Índices monotónicos: {df_players.index.is_monotonic_increasing}")
        
        if isinstance(df_players.index, pd.MultiIndex):
            print(f"Niveles del MultiIndex: {df_players.index.names}")
            print(f"Ejemplo de índice: {df_players.index[0]}")
        else:
            print(f"Primeros 5 índices: {df_players.index[:5].tolist()}")
        
        # Crear subset para testing (últimas 1000 filas para velocidad)
        print_subsection("🎯 CREACIÓN DE SUBSET DE PRUEBA")
        df_test = df_players.tail(1000).copy()
        print(f"✅ Subset creado: {len(df_test):,} filas")
        print(f"   Jugadores en subset: {df_test['player'].nunique()}")
        print(f"   Asistencias promedio: {df_test['assists'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ ERROR en carga de datos: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. INICIALIZACIÓN DEL FEATURE ENGINEER
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("2️⃣ INICIALIZACIÓN DEL FEATURE ENGINEER", "═")
    
    try:
        print("🔧 Inicializando AssistsFeatureEngineer...")
        feature_engineer = AssistsFeatureEngineer(
            teams_df=df_teams,
            players_df=df_players,
            players_quarters_df=df_quarters
        )
        print("✅ Feature Engineer inicializado correctamente")
        
        # Verificar que los DataFrames se pasaron correctamente
        print_subsection("🔍 VERIFICACIÓN DE DATAFRAMES")
        print(f"teams_df: {'✅ Asignado' if feature_engineer.teams_df is not None else '❌ None'}")
        print(f"players_df: {'✅ Asignado' if feature_engineer.players_df is not None else '❌ None'}")
        print(f"players_quarters_df: {'✅ Asignado' if feature_engineer.players_quarters_df is not None else '❌ None'}")
        
    except Exception as e:
        print(f"❌ ERROR en inicialización: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. GENERACIÓN DE FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("3️⃣ GENERACIÓN DE FEATURES", "═")
    
    try:
        print("🎨 Generando todas las features...")
        print("⏱️  Esto puede tomar unos segundos...\n")
        
        # Generar features (retorna DataFrame transformado)
        df_features = feature_engineer.generate_all_features(df_test)
        
        print(f"✅ Features generadas exitosamente!")
        print(f"   Shape original: {df_test.shape}")
        print(f"   Shape con features: {df_features.shape}")
        print(f"   Features agregadas: {df_features.shape[1] - df_test.shape[1]}")
        
        # Verificar índices después de feature engineering
        print_subsection("🔍 VERIFICACIÓN DE ÍNDICES POST-FEATURES")
        print(f"Tipo de índice: {type(df_features.index)}")
        print(f"Índices únicos: {df_features.index.is_unique}")
        print(f"Longitud del índice: {len(df_features.index)}")
        
        if len(df_features) != len(df_test):
            print(f"⚠️  ADVERTENCIA: Cambio en número de filas!")
            print(f"   Antes: {len(df_test)}, Después: {len(df_features)}")
        
    except Exception as e:
        print(f"❌ ERROR en generación de features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. ANÁLISIS DETALLADO POR GRUPO
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("4️⃣ ANÁLISIS DETALLADO POR GRUPO DE FEATURES", "═")
    
    # Definir grupos de features según la arquitectura
    feature_groups = {
        "GRUPO 0: Dataset Features (~5%)": [
            'ast_to_ratio_enhanced',
            'fast_break_playmaking',
            'second_chance_creation',
            'true_shooting_support',
            'efficiency_rating_synergy',
            'defensive_assist_impact'
        ],
        "GRUPO 1: Evolutionary Features (51.98% - DOMINANTE)": [
            'evolutionary_selection_pressure',
            'evolutionary_dominance_index',
            'player_adaptability_score',
            'evolutionary_momentum_predictor',
            'genetic_algorithm_predictor'
        ],
        "GRUPO 2: Positioning Features (6.49%)": [
            'minutes_based_ast_predictor',
            'minutes_expected'
        ],
        "GRUPO 3: Context/Efficiency Features (3.46%)": [
            'home_away_ast_factor',
            'player_offensive_load_pct_l10'
        ],
        "GRUPO 4: Assist Range Features (~2%)": [
            'ast_explosion_potential',
            'extreme_ast_game_freq'
        ],
        "GRUPO 5A: Basic Historical Features": [
            'ast_avg_3g',
            'ast_avg_10g',
            'ast_avg_20g',
            'ast_season_avg',
            'ast_hybrid_predictor',
            'position_index',
            'adaptive_volatility_predictor',
            'evolutionary_fitness',
            'evolutionary_mutation_rate',
            'dynamic_range_predictor',
            'ultra_efficiency_predictor',
            'high_volume_efficiency'
        ],
        "GRUPO 5B: Dataset-Based Features": [
            'assist_momentum_acceleration',
            'efficiency_game_score_impact',
            'fouls_drawn_playmaking',
            'second_chance_playmaking',
            'fast_break_playmaking_enhanced',
            'defensive_rating_impact',
            'offensive_rating_synergy',
            'blocked_attempts_playmaking',
            'plus_minus_momentum'
        ],
        "GRUPO 5C: Elite & Explosive Features": [
            'elite_player_scaling',
            'explosive_game_potential',
            'high_volume_game_indicator',
            'star_player_momentum'
        ],
        "GRUPO 5D: Quarter-Based Features": [
            'quarter_explosive_detection',
            'quarter_consistency_pattern',
            'quarter_momentum_acceleration',
            'elite_quarter_performance',
            'quarter_based_game_projection'
        ],
        "GRUPO 6: Opponent Context Features (~2.8%)": [
            'real_opponent_def_rating',
            'opp_pts_allowed',
            'opponent_adaptation_score'
        ]
    }
    
    # Analizar cada grupo
    all_stats = {}
    problematic_features = []
    
    for group_name, features in feature_groups.items():
        print_subsection(f"📊 {group_name}")
        
        for feature in features:
            stats = analyze_feature_quality(df_features, feature)
            all_stats[feature] = stats
            print_feature_stats(stats, feature)
            
            # Identificar features problemáticas (>50% NaN)
            if stats['exists'] and stats['nan_percentage'] > 50:
                problematic_features.append({
                    'feature': feature,
                    'nan_pct': stats['nan_percentage'],
                    'group': group_name
                })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. RESUMEN DE CALIDAD
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("5️⃣ RESUMEN DE CALIDAD DE FEATURES", "═")
    
    # Contar features por calidad
    excellent = sum(1 for s in all_stats.values() if s['exists'] and s['nan_percentage'] < 10)
    good = sum(1 for s in all_stats.values() if s['exists'] and 10 <= s['nan_percentage'] < 30)
    warning = sum(1 for s in all_stats.values() if s['exists'] and 30 <= s['nan_percentage'] < 50)
    poor = sum(1 for s in all_stats.values() if s['exists'] and s['nan_percentage'] >= 50)
    missing = sum(1 for s in all_stats.values() if not s['exists'])
    
    print(f"📊 DISTRIBUCIÓN DE CALIDAD:")
    print(f"   ✅ Excelente (<10% NaN): {excellent} features")
    print(f"   ⚠️  Bueno (10-30% NaN): {good} features")
    print(f"   🟡 Advertencia (30-50% NaN): {warning} features")
    print(f"   🔴 Pobre (>50% NaN): {poor} features")
    print(f"   ❌ No existe: {missing} features")
    print(f"   📈 TOTAL: {len(all_stats)} features analizadas")
    
    # Features problemáticas
    if problematic_features:
        print_subsection("⚠️  FEATURES CON PROBLEMAS")
        for pf in sorted(problematic_features, key=lambda x: x['nan_pct'], reverse=True):
            print(f"🔴 {pf['feature']}: {pf['nan_pct']:.1f}% NaN")
            print(f"   Grupo: {pf['group']}")
            
            # Sugerencias de solución
            if 'quarter' in pf['feature'].lower():
                print(f"   💡 Causa: Depende de players_quarters_df")
                print(f"   💡 Solución: Verificar disponibilidad de datos por cuarto")
            elif 'opp' in pf['feature'].lower() or 'opponent' in pf['feature'].lower():
                print(f"   💡 Causa: Depende de datos del oponente")
                print(f"   💡 Solución: Verificar columna 'Opp' y teams_df")
            else:
                print(f"   💡 Solución: Verificar disponibilidad de columnas base")
    else:
        print("✅ ¡No hay features con >50% NaN!")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. VERIFICACIÓN DE FALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("6️⃣ VERIFICACIÓN DE FALLBACKS (DEBE SER 0)", "═")
    
    print("🔍 Buscando valores sospechosos que podrían indicar fallbacks...")
    
    # Valores comunes de fallback
    fallback_values = {
        0: "Cero (común fallback)",
        1.0: "Uno (común para ratios)",
        3.0: "Tres (promedio de asistencias)",
        105.0: "105 (defensive rating)",
        110.0: "110 (offensive rating)",
        20.0: "20 (minutos)",
        0.5: "0.5 (ratio genérico)"
    }
    
    suspicious_features = []
    
    for feature, stats in all_stats.items():
        if not stats['exists']:
            continue
        
        series = df_features[feature]
        
        # Verificar valores exactos sospechosos
        for value, description in fallback_values.items():
            exact_count = (series == value).sum()
            if exact_count > len(series) * 0.5:  # Más del 50% con el mismo valor
                suspicious_features.append({
                    'feature': feature,
                    'value': value,
                    'count': exact_count,
                    'percentage': (exact_count / len(series) * 100),
                    'description': description
                })
    
    if suspicious_features:
        print("⚠️  FEATURES CON VALORES SOSPECHOSOS:")
        for sf in suspicious_features:
            print(f"🟡 {sf['feature']}:")
            print(f"   Valor: {sf['value']} ({sf['description']})")
            print(f"   Frecuencia: {sf['count']:,}/{len(df_features):,} ({sf['percentage']:.1f}%)")
            print(f"   ⚠️  Posible fallback hardcoded!")
    else:
        print("✅ ¡No se detectaron fallbacks sospechosos!")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. VERIFICACIÓN DE ÍNDICES Y ALINEACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("7️⃣ VERIFICACIÓN DE ÍNDICES Y ALINEACIÓN", "═")
    
    print("🔍 Verificando alineación de índices...")
    
    # Verificar que los índices se mantienen
    if df_test.index.equals(df_features.index):
        print("✅ Índices perfectamente alineados")
    else:
        print("⚠️  ADVERTENCIA: Índices no coinciden!")
        print(f"   Índices originales: {len(df_test.index)}")
        print(f"   Índices finales: {len(df_features.index)}")
    
    # Verificar que 'assists' (target) NO está presente (diseño anti-leakage)
    if 'assists' in df_features.columns:
        print("❌ ERROR: Columna target 'assists' presente (RIESGO DE LEAKAGE!)")
    else:
        print("✅ Columna target 'assists' NO presente (diseño anti-leakage correcto)")
        print("   ✅ Feature Engineer retorna SOLO features, sin target")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 8. EJEMPLOS DE DATOS
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("8️⃣ EJEMPLOS DE DATOS GENERADOS", "═")
    
    # Seleccionar algunas features clave para mostrar
    key_features = [
        'ast_avg_10g',
        'evolutionary_selection_pressure',
        'minutes_based_ast_predictor',
        'home_away_ast_factor',
        'real_opponent_def_rating',
        'quarter_explosive_detection'
    ]
    
    available_features = [f for f in key_features if f in df_features.columns]
    
    if available_features:
        print("📋 MUESTRA DE DATOS (últimas 10 filas):")
        # Nota: df_features YA NO contiene 'player' porque retorna SOLO features
        print(df_features[available_features].tail(10).to_string())
        print("\n⚠️  NOTA: 'player' NO está en df_features (por diseño anti-leakage)")
    else:
        print("⚠️  No hay features clave disponibles para mostrar")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 9. REPORTE FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("9️⃣ REPORTE FINAL", "═")
    
    print("📊 RESUMEN EJECUTIVO:")
    print(f"   • Dataset: {len(df_features):,} filas, {df_features.shape[1]} columnas")
    print(f"   • Features generadas: {df_features.shape[1] - df_test.shape[1]}")
    print(f"   • Features analizadas: {len(all_stats)}")
    print(f"   • Features excelentes: {excellent} (✅)")
    print(f"   • Features con problemas: {poor} (🔴)")
    print(f"   • Features sospechosas: {len(suspicious_features)} (🟡)")
    
    print("\n🎯 ESTADO GENERAL:")
    if poor == 0 and len(suspicious_features) == 0:
        print("   ✅ EXCELENTE - Todas las features están correctas")
    elif poor <= 2 and len(suspicious_features) <= 2:
        print("   ⚠️  BUENO - Algunas features necesitan atención")
    else:
        print("   🔴 REQUIERE ATENCIÓN - Múltiples features con problemas")
    
    print("\n💡 RECOMENDACIONES:")
    if poor > 0:
        print(f"   1. Revisar {poor} features con >50% NaN")
    if len(suspicious_features) > 0:
        print(f"   2. Verificar {len(suspicious_features)} features con posibles fallbacks")
    if poor == 0 and len(suspicious_features) == 0:
        print("   ✅ No hay recomendaciones - Sistema funcionando correctamente")
    
    print_section("✅ DEBUG COMPLETADO", "═")

if __name__ == "__main__":
    main()

