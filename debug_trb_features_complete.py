"""
DEBUG COMPLETO - FEATURE ENGINEERING DE REBOTES (TRB)
======================================================
Este script verifica TODO el proceso de generación de features para TRB:
- Carga de datos y validación de índices
- Orden de ejecución de métodos
- Cálculo de cada feature con prints detallados
- Validación de que features faltantes FALLEN (no fillna)
- Verificación de NaN vs datos reales
- Features de ejemplo con valores reales

SIN FALLBACKS - Si falla, debe fallar explícitamente
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.trb.features_trb import ReboundsFeatureEngineer

print("=" * 80)
print("🏀 DEBUG COMPLETO - FEATURE ENGINEERING TRB (REBOTES)")
print("=" * 80)
print()

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("📊 PASO 1: CARGANDO DATOS")
print("-" * 80)

data_loader = NBADataLoader()
players_df, teams_df, players_quarters_df, teams_quarters_df = data_loader.load_data(use_quarters=False, force_reload=False)
quarters_df = players_quarters_df  # Para compatibilidad

print(f"✅ Players data cargado: {players_df.shape}")
print(f"✅ Teams data cargado: {teams_df.shape}")
print(f"✅ Quarters data cargado: {quarters_df.shape if quarters_df is not None else 'None'}")
print()

# ============================================================================
# 2. VALIDAR DATOS BÁSICOS
# ============================================================================
print("📊 PASO 2: VALIDANDO DATOS BÁSICOS")
print("-" * 80)

required_cols = ['player', 'Date', 'rebounds', 'minutes', 'Team', 'Opp']
print(f"Columnas requeridas para TRB: {required_cols}")
missing_cols = [col for col in required_cols if col not in players_df.columns]
if missing_cols:
    print(f"❌ ERROR: Columnas faltantes: {missing_cols}")
    sys.exit(1)
else:
    print(f"✅ Todas las columnas requeridas presentes")

print(f"\nColumnas disponibles en players_df ({len(players_df.columns)}):")
print(list(players_df.columns[:20]))  # Mostrar primeras 20
print()

# ============================================================================
# 3. PREPARAR SUBSET DE DATOS PARA DEBUG
# ============================================================================
print("📊 PASO 3: PREPARANDO SUBSET DE DATOS")
print("-" * 80)

# Filtrar jugadores con buenos datos de rebotes
rebounds_avg = players_df.groupby('player')['rebounds'].mean()
top_rebounders = rebounds_avg.nlargest(10).index.tolist()

print(f"Top 10 reboteadores (por promedio de temporada):")
for i, player in enumerate(top_rebounders, 1):
    avg_trb = rebounds_avg[player]
    games = len(players_df[players_df['player'] == player])
    print(f"  {i}. {player}: {avg_trb:.2f} TRB/game ({games} juegos)")

# Tomar jugadores de ejemplo
test_players = top_rebounders[:5]  # Top 5 para debug
print(f"\n🎯 Jugadores seleccionados para debug: {test_players}")

# Filtrar datos
df_test = players_df[players_df['player'].isin(test_players)].copy()
df_test = df_test.sort_values(['player', 'Date']).reset_index(drop=True)

print(f"✅ Datos filtrados: {df_test.shape}")
print(f"   Rango de fechas: {df_test['Date'].min()} a {df_test['Date'].max()}")
print()

# ============================================================================
# 4. VERIFICAR ÍNDICES ANTES DE FEATURE ENGINEERING
# ============================================================================
print("📊 PASO 4: VERIFICANDO ÍNDICES INICIALES")
print("-" * 80)

print(f"Tipo de índice: {type(df_test.index)}")
print(f"Índice actual: {df_test.index[:10].tolist()}")
print(f"¿Es único? {df_test.index.is_unique}")
print(f"¿Tiene nulos? {df_test.index.isna().any()}")

# Verificar duplicados en player-Date
duplicates = df_test.groupby(['player', 'Date']).size()
duplicates = duplicates[duplicates > 1]
if len(duplicates) > 0:
    print(f"⚠️  ADVERTENCIA: {len(duplicates)} combinaciones player-Date duplicadas")
    print(duplicates.head())
else:
    print("✅ No hay duplicados en player-Date")
print()

# ============================================================================
# 5. INICIALIZAR FEATURE ENGINEER
# ============================================================================
print("📊 PASO 5: INICIALIZANDO FEATURE ENGINEER")
print("-" * 80)

feature_engineer = ReboundsFeatureEngineer(
    players_df=players_df,
    teams_df=teams_df,
    players_quarters_df=quarters_df
)

print(f"✅ Feature Engineer inicializado")
print(f"   Target column: rebounds (default)")
print(f"   Windows: {feature_engineer.windows}")
print()

# ============================================================================
# 6. EJECUTAR FEATURE ENGINEERING CON DEBUG DETALLADO
# ============================================================================
print("📊 PASO 6: EJECUTANDO FEATURE ENGINEERING (CON DEBUG)")
print("=" * 80)
print()

# Hacer copia para verificar cambios
df_before = df_test.copy()
initial_cols = set(df_test.columns)

print("🔧 INICIANDO GENERACIÓN DE FEATURES...")
print("-" * 80)

# Ejecutar feature engineering
try:
    feature_names = feature_engineer.generate_all_features(df_test)
    print(f"\n✅ Feature engineering completado exitosamente")
    print(f"   Features generadas: {len(feature_names)}")
    
    # Separar X y y
    y = df_test['rebounds'].copy()
    X = df_test[feature_names].copy()
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
except Exception as e:
    print(f"\n❌ ERROR en feature engineering: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# 7. ANALIZAR FEATURES GENERADAS
# ============================================================================
print("📊 PASO 7: ANALIZANDO FEATURES GENERADAS")
print("=" * 80)

new_cols = set(df_test.columns) - initial_cols
print(f"\n✅ Nuevas columnas creadas: {len(new_cols)}")
print(f"   Columnas iniciales: {len(initial_cols)}")
print(f"   Columnas finales: {len(df_test.columns)}")
print()

# Categorizar features por tipo
feature_categories = {
    'rebounding_history': [],
    'physical_advantage': [],
    'positioning': [],
    'shooting_efficiency': [],
    'opponent_context': [],
    'game_situation': [],
    'elite_rebounding': [],
    'ensemble_critical': [],
    'advanced_momentum': [],
    'otros': []
}

for col in new_cols:
    categorized = False
    for category in feature_categories.keys():
        if category in col.lower() or any(keyword in col.lower() for keyword in [
            'trb' if category == 'rebounding_history' else '',
            'physical' if category == 'physical_advantage' else '',
            'minutes' if category == 'positioning' else '',
            'fga' if category == 'shooting_efficiency' else '',
            'opp' if category == 'opponent_context' else '',
            'elite' if category == 'elite_rebounding' else '',
            'explosion' if category == 'ensemble_critical' else ''
        ]):
            feature_categories[category].append(col)
            categorized = True
            break
    if not categorized:
        feature_categories['otros'].append(col)

print("📋 FEATURES POR CATEGORÍA:")
print("-" * 80)
for category, features in feature_categories.items():
    if features:
        print(f"\n{category.upper()}: {len(features)} features")
        for feat in sorted(features)[:10]:  # Mostrar primeras 10
            print(f"  • {feat}")
        if len(features) > 10:
            print(f"  ... y {len(features) - 10} más")

print()

# ============================================================================
# 8. VERIFICAR DATOS DE EJEMPLO - JUGADOR POR JUGADOR
# ============================================================================
print("📊 PASO 8: VERIFICANDO DATOS DE EJEMPLO POR JUGADOR")
print("=" * 80)

for player in test_players[:3]:  # Primeros 3 jugadores
    print(f"\n{'=' * 80}")
    print(f"🏀 JUGADOR: {player}")
    print(f"{'=' * 80}")
    
    player_data = df_test[df_test['player'] == player].copy()
    print(f"Juegos disponibles: {len(player_data)}")
    
    if len(player_data) < 10:
        print(f"⚠️  ADVERTENCIA: Solo {len(player_data)} juegos, necesita más para features históricas")
        continue
    
    # Tomar juego #15 (debe tener suficiente historia)
    if len(player_data) >= 15:
        idx = player_data.index[14]
        game_data = player_data.loc[idx]
        
        print(f"\n📅 Juego seleccionado (índice {idx}):")
        print(f"   Fecha: {game_data['Date']}")
        print(f"   Oponente: {game_data['Opp']}")
        print(f"   Rebounds (target): {game_data['rebounds']}")
        print(f"   Minutos: {game_data['minutes']}")
        
        # FEATURES DE HISTORIAL
        print(f"\n📊 FEATURES DE HISTORIAL DE REBOTES:")
        print(f"-" * 60)
        hist_features = ['trb_avg_3g', 'trb_avg_5g', 'trb_avg_10g', 'trb_per_minute_5g']
        for feat in hist_features:
            if feat in game_data:
                val = game_data[feat]
                status = "✅" if not pd.isna(val) else "❌"
                print(f"   {status} {feat}: {val:.3f}" if not pd.isna(val) else f"   {status} {feat}: NaN")
            else:
                print(f"   ❌ {feat}: NO EXISTE")
        
        # FEATURES DE VENTAJA FÍSICA
        print(f"\n💪 FEATURES DE VENTAJA FÍSICA:")
        print(f"-" * 60)
        phys_features = ['physical_dominance_index', 'physical_efficiency', 'physical_dominance_momentum']
        for feat in phys_features:
            if feat in game_data:
                val = game_data[feat]
                status = "✅" if not pd.isna(val) else "❌"
                print(f"   {status} {feat}: {val:.3f}" if not pd.isna(val) else f"   {status} {feat}: NaN")
            else:
                print(f"   ❌ {feat}: NO EXISTE")
        
        # FEATURES DE POSICIONAMIENTO
        print(f"\n📍 FEATURES DE POSICIONAMIENTO:")
        print(f"-" * 60)
        pos_features = ['minutes_avg_5g', 'minutes_projected', 'minutes_efficiency_score', 'interior_play_index']
        for feat in pos_features:
            if feat in game_data:
                val = game_data[feat]
                status = "✅" if not pd.isna(val) else "❌"
                print(f"   {status} {feat}: {val:.3f}" if not pd.isna(val) else f"   {status} {feat}: NaN")
            else:
                print(f"   ❌ {feat}: NO EXISTE")
        
        # FEATURES DE CONTEXTO DEL OPONENTE
        print(f"\n🎯 FEATURES DE CONTEXTO DEL OPONENTE:")
        print(f"-" * 60)
        opp_features = ['opp_pace_real', 'opp_defensive_rating_real', 'opp_reb_strength_real', 'defensive_advantage_vs_opp']
        for feat in opp_features:
            if feat in game_data:
                val = game_data[feat]
                status = "✅" if not pd.isna(val) else "❌"
                print(f"   {status} {feat}: {val:.3f}" if not pd.isna(val) else f"   {status} {feat}: NaN")
            else:
                print(f"   ❌ {feat}: NO EXISTE")
        
        # FEATURES ELITE
        print(f"\n⭐ FEATURES DE REBOTEADORES ELITE:")
        print(f"-" * 60)
        elite_features = ['elite_rebounder_tier', 'ultra_physical_dominance', 'elite_rebounding_efficiency', 'elite_rebounding_momentum']
        for feat in elite_features:
            if feat in game_data:
                val = game_data[feat]
                status = "✅" if not pd.isna(val) else "❌"
                print(f"   {status} {feat}: {val:.3f}" if not pd.isna(val) else f"   {status} {feat}: NaN")
            else:
                print(f"   ❌ {feat}: NO EXISTE")

print()

# ============================================================================
# 9. VERIFICAR NaN vs DATOS REALES
# ============================================================================
print("📊 PASO 9: VERIFICANDO NaN vs DATOS REALES")
print("=" * 80)
print()

# Calcular % de NaN por feature
nan_percentages = {}
for col in new_cols:
    if col in df_test.columns:
        nan_pct = df_test[col].isna().sum() / len(df_test) * 100
        nan_percentages[col] = nan_pct

# Mostrar features con más NaN
print("📋 FEATURES CON MÁS NaN (Top 20):")
print("-" * 80)
sorted_nan = sorted(nan_percentages.items(), key=lambda x: x[1], reverse=True)
for feat, pct in sorted_nan[:20]:
    status = "⚠️ " if pct > 50 else "✅"
    print(f"{status} {feat}: {pct:.1f}% NaN")

print()

# Mostrar features con menos NaN (mejor calidad)
print("📋 FEATURES CON MENOS NaN (Top 20):")
print("-" * 80)
for feat, pct in sorted_nan[-20:]:
    status = "✅"
    print(f"{status} {feat}: {pct:.1f}% NaN")

print()

# ============================================================================
# 10. VERIFICAR ÍNDICES FINALES
# ============================================================================
print("📊 PASO 10: VERIFICANDO ÍNDICES FINALES")
print("=" * 80)

print(f"\nX (features):")
print(f"   Shape: {X.shape}")
print(f"   Tipo de índice: {type(X.index)}")
print(f"   ¿Es único? {X.index.is_unique}")
print(f"   Primeros índices: {X.index[:5].tolist()}")

print(f"\ny (target):")
print(f"   Shape: {y.shape}")
print(f"   Tipo de índice: {type(y.index)}")
print(f"   ¿Es único? {y.index.is_unique}")
print(f"   Primeros índices: {y.index[:5].tolist()}")

print(f"\n¿Índices alineados? {(X.index == y.index).all()}")

print()

# ============================================================================
# 11. VERIFICAR QUE NO HAY FALLBACKS (0s artificiales)
# ============================================================================
print("📊 PASO 11: VERIFICANDO QUE NO HAY FALLBACKS")
print("=" * 80)
print()

# Buscar features con valores sospechosos (muchos 0s o 1s)
suspicious_features = []
for col in new_cols:
    if col in df_test.columns:
        # Verificar si hay muchos 0s exactos (más del 30%)
        zeros_pct = (df_test[col] == 0.0).sum() / len(df_test) * 100
        ones_pct = (df_test[col] == 1.0).sum() / len(df_test) * 100
        
        if zeros_pct > 30 and zeros_pct < 99:  # No contar features que naturalmente son 0
            suspicious_features.append((col, zeros_pct, 'zeros'))
        elif ones_pct > 30 and ones_pct < 99:
            suspicious_features.append((col, ones_pct, 'ones'))

if suspicious_features:
    print("⚠️  FEATURES SOSPECHOSAS (muchos 0s o 1s - posibles fallbacks):")
    print("-" * 80)
    for feat, pct, val_type in sorted(suspicious_features, key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {feat}: {pct:.1f}% {val_type}")
else:
    print("✅ No se encontraron features sospechosas con fallbacks evidentes")

print()

# ============================================================================
# 12. ESTADÍSTICAS FINALES
# ============================================================================
print("📊 PASO 12: ESTADÍSTICAS FINALES")
print("=" * 80)
print()

print(f"✅ RESUMEN FINAL:")
print(f"-" * 80)
print(f"   Jugadores procesados: {len(test_players)}")
print(f"   Juegos totales: {len(df_test)}")
print(f"   Features generadas: {len(new_cols)}")
print(f"   Features en X: {X.shape[1]}")
print(f"   Target en y: {len(y)}")
print(f"   Samples finales: {X.shape[0]}")
print()

# Distribución del target
print(f"📊 DISTRIBUCIÓN DEL TARGET (rebounds):")
print(f"-" * 80)
print(f"   Media: {y.mean():.2f}")
print(f"   Std: {y.std():.2f}")
print(f"   Min: {y.min():.2f}")
print(f"   Max: {y.max():.2f}")
print(f"   Mediana: {y.median():.2f}")
print()

# Features más importantes (por varianza)
print(f"📊 FEATURES CON MAYOR VARIANZA (Top 15):")
print(f"-" * 80)
variances = X.var().sort_values(ascending=False)
for feat, var in variances.head(15).items():
    print(f"   {feat}: {var:.3f}")

print()

print("=" * 80)
print("✅ DEBUG COMPLETO FINALIZADO")
print("=" * 80)
print()
print("🎯 PRÓXIMOS PASOS:")
print("   1. Revisar features con muchos NaN (si los hay)")
print("   2. Revisar features sospechosas (si las hay)")
print("   3. Verificar que los datos de ejemplo son correctos")
print("   4. Si todo está OK, entrenar el modelo con estos datos")
print()

