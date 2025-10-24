"""
Investigar por qué _get_historical_series retorna 100% NaN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.ast.features_ast import AssistsFeatureEngineer
import pandas as pd

# Cargar datos
print("Cargando datos...")
data_loader = NBADataLoader()
result = data_loader.load_data(use_quarters=False, force_reload=False)
if isinstance(result, tuple):
    df_players, df_teams, _, _ = result
else:
    df_players = result
    df_teams = None

print(f"Dataset completo: {len(df_players):,} filas")
print(f"Jugadores únicos: {df_players['player'].nunique()}")

# Probar con subset
subset = df_players.tail(1000).copy()
print(f"\nSubset (últimas 1000 filas): {len(subset)} filas")
print(f"Jugadores en subset: {subset['player'].nunique()}")
print(f"Jugadores: {subset['player'].unique()[:5]}")

# Verificar columna assists
print(f"\n'assists' en columnas: {'assists' in subset.columns}")
print(f"Assists stats:\n{subset['assists'].describe()}")

# Probar _get_historical_series manualmente
print("\n" + "="*80)
print("PRUEBA MANUAL DE _get_historical_series")
print("="*80)

feature_engineer = AssistsFeatureEngineer(
    teams_df=df_teams,
    players_df=df_players,
    players_quarters_df=None
)

# Probar con ventana de 10 juegos
print("\nProbando _get_historical_series(df, 'assists', window=10, operation='mean')...")
result_series = feature_engineer._get_historical_series(subset, 'assists', window=10, operation='mean')

print(f"Tipo de resultado: {type(result_series)}")
print(f"Longitud: {len(result_series)}")
print(f"NaN count: {result_series.isna().sum()}")
print(f"Valid count: {result_series.notna().sum()}")
print(f"NaN percentage: {result_series.isna().sum() / len(result_series) * 100:.1f}%")

if result_series.notna().sum() > 0:
    print(f"\nEstadísticas de valores válidos:")
    print(result_series.describe())
    print(f"\nPrimeros 20 valores:")
    print(result_series.head(20))
else:
    print("\n❌ TODOS LOS VALORES SON NaN!")
    print("\nInvestigando causa...")
    
    # Verificar si hay datos por jugador
    print("\nDatos por jugador:")
    for player in subset['player'].unique()[:3]:
        player_data = subset[subset['player'] == player]
        print(f"\n{player}:")
        print(f"  Total juegos: {len(player_data)}")
        print(f"  Assists: {player_data['assists'].tolist()}")
        
        # Intentar calcular rolling manualmente
        if len(player_data) >= 10:
            rolling_result = player_data['assists'].rolling(window=10, min_periods=1).mean().shift(1)
            print(f"  Rolling mean (10g): {rolling_result.tolist()}")
        else:
            print(f"  ⚠️  Menos de 10 juegos!")

print("\n" + "="*80)
print("Probando con un jugador específico con muchos juegos...")
print("="*80)

# Encontrar jugador con más juegos
player_counts = df_players['player'].value_counts()
top_player = player_counts.index[0]
top_player_games = player_counts.iloc[0]

print(f"\nJugador con más juegos: {top_player} ({top_player_games} juegos)")

# Tomar últimos 100 juegos de ese jugador
player_data = df_players[df_players['player'] == top_player].tail(100).copy()
print(f"Tomando últimos 100 juegos de {top_player}")

# Probar _get_historical_series
result_series2 = feature_engineer._get_historical_series(player_data, 'assists', window=10, operation='mean')
print(f"\nResultado:")
print(f"  NaN count: {result_series2.isna().sum()}")
print(f"  Valid count: {result_series2.notna().sum()}")
print(f"  NaN percentage: {result_series2.isna().sum() / len(result_series2) * 100:.1f}%")

if result_series2.notna().sum() > 0:
    print(f"\n✅ ¡Funciona con un solo jugador!")
    print(f"Estadísticas:")
    print(result_series2.describe())
else:
    print(f"\n❌ Aún retorna 100% NaN incluso con un solo jugador!")

